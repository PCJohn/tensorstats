#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define TS_RESTRICT __restrict
#else
#define TS_RESTRICT __restrict__
#endif

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// tensorstats — fast exact central moment computation
//
// Output convention: moments-LAST
//   global reduction  → (n_moments,)
//   per-axis          → (*kept_shape, n_moments)
//   grid              → (*cell_shape, n_moments)
//
// Moments layout: [0]=mean  [1]=variance  [2]=m3  [3]=m4
//
// Computation paths (selected at runtime per axis spec):
//   Global    — straight loop over all elements, AVX2-friendly
//   LastAxis  — reducing all axes but the last (e.g. axes=(0,1) on HxWxC).
//               Contiguous: column loops without modulo. Strided: a single
//               walk over the leading axes collects all channels at once
//               (last_axis_fused), keeping uint8 on the histogram path.
//   General   — iterative (flat_idx, bucket) pairs for arbitrary axes
//   Grid      — pixel-outer scatter into per-cell accumulators;
//               precomputed int16 cell_of[] eliminates per-pixel arithmetic
//
// uint8 fast path (Global + LastAxis):
//   Builds hist[256], computes moments with 256 FMA operations instead of N.
//   4-way parallel counters reduce write-after-write stalls.
// ---------------------------------------------------------------------------

// Number of histogram bins — always 256 for uint8.
static constexpr int HIST_BINS = 256;

// Minimum pixels-per-cell to prefer the histogram path in the grid.
static constexpr int64_t HIST_THRESHOLD = 256;

// Maximum number of grid cells (must fit in int16_t).
static constexpr int64_t MAX_GRID_CELLS = 32767;

// ---------------------------------------------------------------------------
// uint8 histogram helpers
// ---------------------------------------------------------------------------

static void build_hist4(const uint8_t *TS_RESTRICT d, int64_t n,
                        int64_t *TS_RESTRICT hist) {
  int64_t h0[HIST_BINS] = {}, h1[HIST_BINS] = {};
  int64_t h2[HIST_BINS] = {}, h3[HIST_BINS] = {};
  int64_t i = 0, n4 = (n >> 2) << 2;
  for (; i < n4; i += 4) {
    h0[d[i]]++;
    h1[d[i + 1]]++;
    h2[d[i + 2]]++;
    h3[d[i + 3]]++;
  }
  for (; i < n; ++i)
    h0[d[i]]++;
  for (int v = 0; v < HIST_BINS; ++v)
    hist[v] = h0[v] + h1[v] + h2[v] + h3[v];
}

static void moments_from_hist(const int64_t *TS_RESTRICT hist, int64_t n,
                              double &mu, double &m2, double &m3, double &m4) {
  int64_t s1 = 0;
  for (int v = 0; v < HIST_BINS; ++v)
    s1 += hist[v] * (int64_t)v;
  mu = (double)s1 / (double)n;
  m2 = m3 = m4 = 0.0;
  for (int v = 0; v < HIST_BINS; ++v) {
    if (!hist[v])
      continue;
    double x = (double)v - mu, x2 = x * x, h = (double)hist[v];
    m2 += h * x2;
    m3 += h * x2 * x;
    m4 += h * x2 * x2;
  }
  const double inv = 1.0 / (double)n;
  m2 *= inv;
  m3 *= inv;
  m4 *= inv;
}

// Compute global moments from uint8 data with optional uniform stride.
static void global_u8_hist(const uint8_t *TS_RESTRICT d, int64_t n,
                           int64_t step, double &mu, double &m2, double &m3,
                           double &m4) {
  int64_t hist[HIST_BINS] = {};
  if (step == 1) {
    build_hist4(d, n, hist);
    moments_from_hist(hist, n, mu, m2, m3, m4);
  } else {
    int64_t h0[HIST_BINS] = {}, h1[HIST_BINS] = {};
    int64_t h2[HIST_BINS] = {}, h3[HIST_BINS] = {};
    const int64_t step4 = step * 4;
    int64_t ns = 0, i = 0, n4 = (n / step4) * step4;
    for (; i < n4; i += step4) {
      h0[d[i]]++;
      h1[d[i + step]]++;
      h2[d[i + step * 2]]++;
      h3[d[i + step * 3]]++;
      ns += 4;
    }
    for (; i < n; i += step) {
      h0[d[i]]++;
      ns++;
    }
    for (int v = 0; v < HIST_BINS; ++v)
      hist[v] = h0[v] + h1[v] + h2[v] + h3[v];
    moments_from_hist(hist, ns, mu, m2, m3, m4);
  }
}

// Compute per-channel moments from uint8 data, reducing over HW rows.
static void last_axis_u8_hist(const uint8_t *TS_RESTRICT d, int64_t HW,
                              int64_t C, int64_t sr, int64_t sc,
                              std::vector<double> &mu, std::vector<double> &m2,
                              std::vector<double> &m3,
                              std::vector<double> &m4) {
  for (int64_t c = 0; c < C; c += sc) {
    int64_t hist[HIST_BINS] = {}, ns = 0;
    for (int64_t r = 0; r < HW; r += sr) {
      hist[d[r * C + c]]++;
      ns++;
    }
    moments_from_hist(hist, ns, mu[c], m2[c], m3[c], m4[c]);
  }
}

// ---------------------------------------------------------------------------
// Sampled flat index list — used only for non-uniform strides
// ---------------------------------------------------------------------------
static std::vector<int64_t>
sampled_indices(const std::vector<int64_t> &shape,
                const std::vector<int64_t> &stride) {
  const int ndim = (int)shape.size();
  int64_t cap = 1;
  for (int d = 0; d < ndim; ++d)
    cap *= (shape[d] + stride[d] - 1) / stride[d];

  std::vector<int64_t> fs(ndim, 1);
  for (int d = ndim - 2; d >= 0; --d)
    fs[d] = fs[d + 1] * shape[d + 1];

  std::vector<int64_t> result, coords(ndim, 0);
  result.reserve(cap);
  int64_t flat = 0;
  while (true) {
    result.push_back(flat);
    int d = ndim - 1;
    while (d >= 0) {
      coords[d] += stride[d];
      flat += stride[d] * fs[d];
      if (coords[d] < shape[d])
        break;
      flat -= coords[d] * fs[d];
      coords[d] = 0;
      --d;
    }
    if (d < 0)
      break;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Inner loop kernels
// ---------------------------------------------------------------------------

template <typename T>
static void global_pass(const T *TS_RESTRICT data, int64_t n, int64_t step,
                        double &mu, double &m2, double &m3, double &m4) {
  const int64_t ns = (n + step - 1) / step;
  double s = 0;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
  for (int64_t i = 0; i < n; i += step)
    s += (double)data[i];
  mu = s / (double)ns;
  m2 = m3 = m4 = 0;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
  for (int64_t i = 0; i < n; i += step) {
    const double d = (double)data[i] - mu, d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  const double inv = 1.0 / (double)ns;
  m2 *= inv;
  m3 *= inv;
  m4 *= inv;
}

template <typename T>
static void global_pass_idx(const T *TS_RESTRICT data,
                            const std::vector<int64_t> &idx, double &mu,
                            double &m2, double &m3, double &m4) {
  const int64_t ns = (int64_t)idx.size();
  double s = 0;
  for (int64_t i : idx)
    s += (double)data[i];
  mu = s / (double)ns;
  m2 = m3 = m4 = 0;
  for (int64_t i : idx) {
    const double d = (double)data[i] - mu, d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  const double inv = 1.0 / (double)ns;
  m2 *= inv;
  m3 *= inv;
  m4 *= inv;
}

template <typename T>
static void last_axis_pass(const T *TS_RESTRICT data, int64_t HW, int64_t C,
                           int64_t sr, int64_t sc, std::vector<double> &mu,
                           std::vector<double> &m2, std::vector<double> &m3,
                           std::vector<double> &m4) {
  const int64_t nrows = (HW + sr - 1) / sr;
  const double inv = 1.0 / (double)nrows;
  for (int64_t c = 0; c < C; c += sc) {
    double s = 0;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (int64_t r = 0; r < HW; r += sr)
      s += (double)data[r * C + c];
    mu[c] = s * inv;
    double s2 = 0, s3 = 0, s4 = 0;
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (int64_t r = 0; r < HW; r += sr) {
      const double d = (double)data[r * C + c] - mu[c], d2 = d * d;
      s2 += d2;
      s3 += d2 * d;
      s4 += d2 * d2;
    }
    m2[c] = s2 * inv;
    m3[c] = s3 * inv;
    m4[c] = s4 * inv;
  }
}

template <typename T>
static void general_pass(const T *TS_RESTRICT data,
                         const std::vector<std::pair<int64_t, int64_t>> &pairs,
                         int64_t n_buckets, std::vector<double> &mu,
                         std::vector<double> &m2, std::vector<double> &m3,
                         std::vector<double> &m4) {
  std::fill(mu.begin(), mu.end(), 0.0);
  std::vector<int64_t> counts(n_buckets, 0);
  for (const auto &[fi, b] : pairs) {
    mu[b] += (double)data[fi];
    counts[b]++;
  }
  for (int64_t b = 0; b < n_buckets; ++b)
    mu[b] = counts[b] > 0 ? mu[b] / (double)counts[b] : 0.0;
  std::fill(m2.begin(), m2.end(), 0.0);
  std::fill(m3.begin(), m3.end(), 0.0);
  std::fill(m4.begin(), m4.end(), 0.0);
  for (const auto &[fi, b] : pairs) {
    const double d = (double)data[fi] - mu[b], d2 = d * d;
    m2[b] += d2;
    m3[b] += d2 * d;
    m4[b] += d2 * d2;
  }
  for (int64_t b = 0; b < n_buckets; ++b)
    if (counts[b] > 0) {
      const double inv = 1.0 / (double)counts[b];
      m2[b] *= inv;
      m3[b] *= inv;
      m4[b] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Fused last-axis (per-channel) reduction with arbitrary per-axis stride.
//
// When all axes but the last are reduced, the last axis survives as the output
// "channels" and a row's C channel values sit contiguously at base..base+C-1.
// A single strided walk over the leading axes therefore collects every channel
// at once — no per-channel rescan — and for uint8 the histogram fast path keeps
// working even when several leading axes are strided (the case the general path
// handles only by gathering, losing the histogram entirely).
// ---------------------------------------------------------------------------

// Walk each leading-axis position (all axes but the last) honoring per-axis
// stride, invoking fn(base) with the flat offset of its first channel. Returns
// the number of rows visited. The last axis is assumed contiguous (stride 1).
template <typename Fn>
static int64_t for_each_row(const std::vector<int64_t> &shape,
                            const std::vector<int64_t> &stride, Fn &&fn) {
  const int ndim = (int)shape.size(), lead = ndim - 1;
  std::vector<int64_t> fstride(ndim, 1);
  for (int d = ndim - 2; d >= 0; --d)
    fstride[d] = fstride[d + 1] * shape[d + 1];
  std::vector<int64_t> coord(lead, 0);
  int64_t base = 0, nrows = 0;
  while (true) {
    fn(base);
    ++nrows;
    int d = lead - 1;
    for (; d >= 0; --d) {
      coord[d] += stride[d];
      base += stride[d] * fstride[d];
      if (coord[d] < shape[d])
        break;
      base -= coord[d] * fstride[d];
      coord[d] = 0;
    }
    if (d < 0)
      break;
  }
  return nrows;
}

// uint8: one histogram per channel, filled in a single strided walk.
static void last_axis_u8_fused(const uint8_t *TS_RESTRICT d,
                               const std::vector<int64_t> &shape,
                               const std::vector<int64_t> &stride, int64_t C,
                               std::vector<double> &mu, std::vector<double> &m2,
                               std::vector<double> &m3,
                               std::vector<double> &m4) {
  std::vector<int64_t> hist((size_t)C * HIST_BINS, 0);
  int64_t *TS_RESTRICT h = hist.data();
  const int64_t nrows = for_each_row(shape, stride, [&](int64_t base) {
    for (int64_t c = 0; c < C; ++c)
      h[c * HIST_BINS + d[base + c]]++;
  });
  for (int64_t c = 0; c < C; ++c)
    moments_from_hist(&h[c * HIST_BINS], nrows, mu[c], m2[c], m3[c], m4[c]);
}

// float: per-channel two-pass (mean, then central moments) over the same walk.
template <typename T>
static void last_axis_fused(const T *TS_RESTRICT d,
                            const std::vector<int64_t> &shape,
                            const std::vector<int64_t> &stride, int64_t C,
                            std::vector<double> &mu, std::vector<double> &m2,
                            std::vector<double> &m3, std::vector<double> &m4) {
  std::fill(mu.begin(), mu.end(), 0.0);
  const int64_t nrows = for_each_row(shape, stride, [&](int64_t base) {
    for (int64_t c = 0; c < C; ++c)
      mu[c] += (double)d[base + c];
  });
  const double inv = 1.0 / (double)nrows;
  for (int64_t c = 0; c < C; ++c)
    mu[c] *= inv;
  std::fill(m2.begin(), m2.end(), 0.0);
  std::fill(m3.begin(), m3.end(), 0.0);
  std::fill(m4.begin(), m4.end(), 0.0);
  for_each_row(shape, stride, [&](int64_t base) {
    for (int64_t c = 0; c < C; ++c) {
      const double x = (double)d[base + c] - mu[c], x2 = x * x;
      m2[c] += x2;
      m3[c] += x2 * x;
      m4[c] += x2 * x2;
    }
  });
  for (int64_t c = 0; c < C; ++c) {
    m2[c] *= inv;
    m3[c] *= inv;
    m4[c] *= inv;
  }
}

// ---------------------------------------------------------------------------
// Axis spec — describes one reduction (which axes to reduce, output shape)
// ---------------------------------------------------------------------------
struct AxisSpec {
  std::vector<int> reduce_axes;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> out_strides;
  int64_t acc_size = 1;
  bool is_last_dim = false; // true when reducing all axes except the last
  int64_t last_dim_C = 1;   // size of the last (kept) axis when is_last_dim
};

static AxisSpec make_axis_spec(const std::vector<int> &axes,
                               const std::vector<int64_t> &shape) {
  const int ndim = (int)shape.size();
  AxisSpec s;
  for (int a : axes) {
    int na = (a < 0) ? a + ndim : a;
    if (na < 0 || na >= ndim)
      throw std::out_of_range("axis out of range");
    s.reduce_axes.push_back(na);
  }
  std::sort(s.reduce_axes.begin(), s.reduce_axes.end());
  s.reduce_axes.erase(std::unique(s.reduce_axes.begin(), s.reduce_axes.end()),
                      s.reduce_axes.end());
  s.acc_size = 1;
  for (int d = 0; d < ndim; ++d) {
    const bool red =
        std::binary_search(s.reduce_axes.begin(), s.reduce_axes.end(), d);
    if (!red) {
      s.out_shape.push_back(shape[d]);
      s.acc_size *= shape[d];
    }
  }
  s.out_strides.resize(s.out_shape.size(), 1);
  for (int i = (int)s.out_shape.size() - 2; i >= 0; --i)
    s.out_strides[i] = s.out_strides[i + 1] * s.out_shape[i + 1];

  // is_last_dim: axes are exactly [0, 1, ..., ndim-2]
  if ((int)s.reduce_axes.size() == ndim - 1) {
    bool contiguous = true;
    for (int i = 0; i < ndim - 1; ++i)
      if (s.reduce_axes[i] != i) {
        contiguous = false;
        break;
      }
    if (contiguous) {
      s.is_last_dim = true;
      s.last_dim_C = shape[ndim - 1];
    }
  }
  return s;
}

// Build (flat_idx, bucket_idx) pairs for the general reduction path.
// Fully iterative — no std::function overhead.
static std::vector<std::pair<int64_t, int64_t>>
make_pairs(const std::vector<int64_t> &shape,
           const std::vector<int64_t> &stride,
           const std::vector<int> &reduce_axes,
           const std::vector<int64_t> &out_strides) {
  const int ndim = (int)shape.size();
  std::vector<bool> is_red(ndim, false);
  std::vector<int64_t> odim(ndim, -1), fs(ndim, 1);
  for (int a : reduce_axes)
    is_red[a] = true;
  int od = 0;
  for (int d = 0; d < ndim; ++d)
    if (!is_red[d])
      odim[d] = od++;
  for (int d = ndim - 2; d >= 0; --d)
    fs[d] = fs[d + 1] * shape[d + 1];

  int64_t cap = 1;
  for (int d = 0; d < ndim; ++d)
    cap *= (shape[d] + stride[d] - 1) / stride[d];
  std::vector<std::pair<int64_t, int64_t>> pairs;
  pairs.reserve(cap);

  std::vector<int64_t> coords(ndim, 0);
  int64_t flat = 0, bkt = 0;
  while (true) {
    pairs.push_back({flat, bkt});
    int d = ndim - 1;
    while (d >= 0) {
      coords[d] += stride[d];
      flat += stride[d] * fs[d];
      if (!is_red[d] && odim[d] >= 0)
        bkt += stride[d] * out_strides[odim[d]];
      if (coords[d] < shape[d])
        break;
      flat -= coords[d] * fs[d];
      if (!is_red[d] && odim[d] >= 0)
        bkt -= coords[d] * out_strides[odim[d]];
      coords[d] = 0;
      --d;
    }
    if (d < 0)
      break;
  }
  return pairs;
}

// ---------------------------------------------------------------------------
// Pack — write moment vectors into a new heap-allocated ndarray.
//        Used by the axes-reduction path (grid uses a retained buffer instead).
// ---------------------------------------------------------------------------
static nb::ndarray<nb::numpy, double>
pack(const std::vector<double> &mu, const std::vector<double> &m2,
     const std::vector<double> &m3, const std::vector<double> &m4,
     const std::vector<int64_t> &out_shape, int64_t nacc, int n_moments) {
  std::vector<size_t> sh(out_shape.begin(), out_shape.end());
  sh.push_back((size_t)n_moments);
  auto *ptr = new double[(size_t)nacc * (size_t)n_moments];
  for (int64_t i = 0; i < nacc; ++i) {
    if (n_moments >= 1)
      ptr[i * n_moments + 0] = mu[i];
    if (n_moments >= 2)
      ptr[i * n_moments + 1] = m2[i];
    if (n_moments >= 3)
      ptr[i * n_moments + 2] = m3[i];
    if (n_moments >= 4)
      ptr[i * n_moments + 3] = m4[i];
  }
  nb::capsule own(ptr,
                  [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double>(ptr, sh.size(), sh.data(), own);
}

// ---------------------------------------------------------------------------
// Axes reduction entry point — handles global, per-axis, and general reductions
// ---------------------------------------------------------------------------
template <typename T>
static nb::dict compute_typed(const T *data, const std::vector<int64_t> &shape,
                              const std::vector<std::vector<int>> &axes_list,
                              const std::vector<int64_t> &stride,
                              int n_moments) {
  const int ndim = (int)shape.size();
  const int64_t total = std::accumulate(shape.begin(), shape.end(), (int64_t)1,
                                        std::multiplies<int64_t>{});

  // Classify stride in one pass: uniform (all equal) vs non-uniform vs none.
  const int64_t s0 = stride[0];
  bool uniform_stride = std::all_of(stride.begin(), stride.end(),
                                    [s0](int64_t s) { return s == s0; });
  bool has_stride = std::any_of(stride.begin(), stride.end(),
                                [](int64_t s) { return s > 1; });

  nb::dict result;

  for (const auto &raw_axes : axes_list) {
    if (raw_axes.empty()) {
      // --- Global reduction ---
      double mu = 0, m2 = 0, m3 = 0, m4 = 0;
      if constexpr (std::is_same_v<T, uint8_t>) {
        if (!has_stride || uniform_stride)
          global_u8_hist(data, total, uniform_stride ? s0 : 1, mu, m2, m3, m4);
        else
          global_pass_idx<T>(data, sampled_indices(shape, stride), mu, m2, m3,
                             m4);
      } else {
        if (!has_stride)
          global_pass<T>(data, total, 1, mu, m2, m3, m4);
        else if (uniform_stride)
          global_pass<T>(data, total, s0, mu, m2, m3, m4);
        else
          global_pass_idx<T>(data, sampled_indices(shape, stride), mu, m2, m3,
                             m4);
      }
      auto *p = new double[n_moments];
      if (n_moments >= 1)
        p[0] = mu;
      if (n_moments >= 2)
        p[1] = m2;
      if (n_moments >= 3)
        p[2] = m3;
      if (n_moments >= 4)
        p[3] = m4;
      const size_t sh = (size_t)n_moments;
      nb::capsule own(
          p, [](void *x) noexcept { delete[] static_cast<double *>(x); });
      result["global"] = nb::ndarray<nb::numpy, double>(p, 1, &sh, own);
    } else {
      // --- Per-axis reduction ---
      const AxisSpec spec = make_axis_spec(raw_axes, shape);
      const int64_t nacc = spec.acc_size;
      std::vector<double> mu(nacc), m2(nacc), m3(nacc), m4(nacc);

      if (spec.is_last_dim) {
        // Reducing all axes except the last (e.g. axes=(0,1) on HxWxC). With
        // any stride, last_axis_fused walks the leading axes once and collects
        // all channels — it handles single- and multi-axis stride uniformly, so
        // uint8 keeps its histogram path instead of falling back to a gather.
        // A strided last (channel) axis can't be collected contiguously, so
        // that rarer case still goes through general_pass.
        const int64_t HW = total / spec.last_dim_C;
        const int64_t C = spec.last_dim_C;
        const bool channel_contig = stride[ndim - 1] == 1;

        if constexpr (std::is_same_v<T, uint8_t>) {
          if (!has_stride)
            last_axis_u8_hist(data, HW, C, 1, 1, mu, m2, m3, m4);
          else if (channel_contig)
            last_axis_u8_fused(data, shape, stride, C, mu, m2, m3, m4);
          else
            general_pass<T>(
                data,
                make_pairs(shape, stride, spec.reduce_axes, spec.out_strides),
                nacc, mu, m2, m3, m4);
        } else {
          if (!has_stride)
            last_axis_pass<T>(data, HW, C, 1, 1, mu, m2, m3, m4);
          else if (channel_contig)
            last_axis_fused<T>(data, shape, stride, C, mu, m2, m3, m4);
          else
            general_pass<T>(
                data,
                make_pairs(shape, stride, spec.reduce_axes, spec.out_strides),
                nacc, mu, m2, m3, m4);
        }
      } else {
        // General path: arbitrary axes
        general_pass<T>(
            data, make_pairs(shape, stride, spec.reduce_axes, spec.out_strides),
            nacc, mu, m2, m3, m4);
      }

      // Build result dict key e.g. "0,1" for axes=(0,1)
      std::string key;
      for (int i = 0; i < (int)raw_axes.size(); ++i) {
        if (i)
          key += ',';
        key += std::to_string(raw_axes[i]);
      }
      result[key.c_str()] =
          pack(mu, m2, m3, m4, spec.out_shape, nacc, n_moments);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// _GridStatsComputerImpl — internal stateful multi-grid moment computer.
//
// Not exposed publicly. Used by StatsComputer (Python) via nanobind.
//
// Computes exact central moments [mean, variance, m3, m4] per grid cell, for
// one OR MORE grid resolutions at once (a "grid pyramid"). A single grid is the
// K=1 degenerate case of the same generic path — there is no separate K=1 code.
//
// Single-sweep fusion: pixels are visited once per pass and scattered into
// every level's accumulators (pixel-outer, level-inner). Adding levels adds
// accumulators, not tensor reads — the tensor is read once when every level
// uses the uint8 histogram path, twice when any level needs the direct two-pass
// (the inherent cost of exact central moments). It is never read K times.
//
// Optional stride subsamples the input uniformly; each level then holds exact
// moments over the visited (subsampled) pixels of its cells. The strided visit
// order is precomputed once at config time (sampled_).
//
// Retained across calls (fixed shape + grid config), per level:
//   cell_of   int16 flat cell index per pixel — precomputed, no per-pixel
//   arith. mu/m2/m3/m4/counts  accumulators (total_cells each). hists per-cell
//   hist[256] (uint8 histogram path only). out_buf   output buffer (total_cells
//   * n_moments). compute_* returns VIEWS;
//             Python copies before the next call.
//
// Grid cell assignment (per level, per axis): floor-division gives uniform
// boundaries for any shape — cell[coord,d] = coord * n_cells[d] / shape[d].
// ---------------------------------------------------------------------------

// Per-grid-level retained state. One instance per requested grid resolution.
struct GridLevel {
  std::vector<size_t> out_shape; // (*n_cells, n_moments)
  int64_t total_cells = 0;
  bool use_hist = false;        // uint8 histogram path (decided at config)
  std::vector<int16_t> cell_of; // per pixel (full flat index) -> cell
  std::vector<double> mu, m2, m3, m4, out_buf;
  std::vector<int64_t> counts;
  std::vector<int64_t> hists; // total_cells * HIST_BINS (use_hist only)
};

// Finalise one cell's central moments into out_buf. Empty cells (possible once
// stride, or n_cells > shape, leaves a cell with no visited pixels) are zeroed
// rather than left holding stale data from a previous call.
static inline void pack_cell(GridLevel &L, int64_t c, int nm) {
  double *out = L.out_buf.data() + c * nm;
  if (!L.counts[c]) {
    for (int k = 0; k < nm; ++k)
      out[k] = 0.0;
    return;
  }
  const double inv = 1.0 / (double)L.counts[c];
  if (nm >= 1)
    out[0] = L.mu[c];
  if (nm >= 2)
    out[1] = L.m2[c] * inv;
  if (nm >= 3)
    out[2] = L.m3[c] * inv;
  if (nm >= 4)
    out[3] = L.m4[c] * inv;
}

class _GridStatsComputerImpl {
  int64_t total_ = 0; // total pixels (product of shape)
  int n_moments_ = 4;
  bool strided_ = false;
  std::vector<int64_t> sampled_; // strided visit order (empty if !strided_)
  std::vector<GridLevel> levels_;
  std::vector<int> hist_idx_,
      direct_idx_; // partition of levels_ for the uint8 path

  nb::ndarray<nb::numpy, double> _view(GridLevel &L) {
    return nb::ndarray<nb::numpy, double>(L.out_buf.data(), L.out_shape.size(),
                                          L.out_shape.data(), nb::none());
  }

  nb::list _views() {
    nb::list out;
    for (GridLevel &L : levels_)
      out.append(_view(L));
    return out;
  }

  // Build one grid level from a per-axis cell-exponent vector (grid[d]=k ->
  // 2^k).
  GridLevel _make_level(const std::vector<int64_t> &shape,
                        const std::vector<int> &grid) {
    const int ndim = (int)shape.size();
    GridLevel L;
    std::vector<int64_t> n_cells(ndim), cs(ndim, 1);
    for (int d = 0; d < ndim; ++d)
      n_cells[d] = (int64_t)1 << grid[d];
    L.total_cells = std::accumulate(n_cells.begin(), n_cells.end(), (int64_t)1,
                                    std::multiplies<int64_t>{});
    if (L.total_cells > MAX_GRID_CELLS)
      throw std::runtime_error(
          "_GridStatsComputerImpl: total_cells exceeds int16 range");
    for (int d = ndim - 2; d >= 0; --d)
      cs[d] = cs[d + 1] * n_cells[d + 1];

    L.use_hist =
        (L.total_cells > 0) && (total_ / L.total_cells >= HIST_THRESHOLD);

    // Per-axis cell LUTs flattened to cell_of[pixel].
    std::vector<std::vector<int64_t>> lut(ndim);
    for (int d = 0; d < ndim; ++d) {
      lut[d].resize(shape[d]);
      for (int64_t i = 0; i < shape[d]; ++i)
        lut[d][i] = i * n_cells[d] / shape[d];
    }
    L.cell_of.resize(total_);
    std::vector<int64_t> coords(ndim, 0);
    for (int64_t flat = 0; flat < total_; ++flat) {
      int64_t cell = 0;
      for (int d = 0; d < ndim; ++d)
        cell += lut[d][coords[d]] * cs[d];
      L.cell_of[flat] = (int16_t)cell;
      for (int d = ndim - 1; d >= 0; --d) {
        if (++coords[d] < shape[d])
          break;
        coords[d] = 0;
      }
    }

    L.mu.assign(L.total_cells, 0.0);
    L.m2.assign(L.total_cells, 0.0);
    L.m3.assign(L.total_cells, 0.0);
    L.m4.assign(L.total_cells, 0.0);
    L.counts.assign(L.total_cells, 0);
    if (L.use_hist)
      L.hists.assign(L.total_cells * HIST_BINS, 0);
    L.out_buf.assign(L.total_cells * n_moments_, 0.0);

    L.out_shape.assign(n_cells.begin(), n_cells.end());
    L.out_shape.push_back((size_t)n_moments_);
    return L;
  }

  // uint8 fused sweep. Hist levels finalise from their per-cell histogram in
  // one pass; direct levels use the two-pass (sum -> mean, then central
  // moments).
  template <bool Strided>
  void _run_u8(const uint8_t *TS_RESTRICT data, int64_t nv) {
    const int nm = n_moments_;
    for (int li : hist_idx_) {
      std::fill(levels_[li].hists.begin(), levels_[li].hists.end(), 0);
      std::fill(levels_[li].counts.begin(), levels_[li].counts.end(), 0);
    }
    for (int li : direct_idx_) {
      std::fill(levels_[li].mu.begin(), levels_[li].mu.end(), 0.0);
      std::fill(levels_[li].counts.begin(), levels_[li].counts.end(), 0);
    }
    // Pass 1: histograms (hist levels) + sums (direct levels), single sweep.
    for (int64_t k = 0; k < nv; ++k) {
      const int64_t i = Strided ? sampled_[k] : k;
      const uint8_t v = data[i];
      for (int li : hist_idx_) {
        GridLevel &L = levels_[li];
        const int16_t c = L.cell_of[i];
        L.hists[(int64_t)c * HIST_BINS + v]++;
        L.counts[c]++;
      }
      for (int li : direct_idx_) {
        GridLevel &L = levels_[li];
        const int16_t c = L.cell_of[i];
        L.mu[c] += (double)v;
        L.counts[c]++;
      }
    }
    for (int li : hist_idx_) {
      GridLevel &L = levels_[li];
      for (int64_t c = 0; c < L.total_cells; ++c) {
        double *out = L.out_buf.data() + c * nm;
        if (!L.counts[c]) {
          for (int q = 0; q < nm; ++q)
            out[q] = 0.0;
          continue;
        }
        double mu, m2, m3, m4;
        moments_from_hist(L.hists.data() + c * HIST_BINS, L.counts[c], mu, m2,
                          m3, m4);
        if (nm >= 1)
          out[0] = mu;
        if (nm >= 2)
          out[1] = m2;
        if (nm >= 3)
          out[2] = m3;
        if (nm >= 4)
          out[3] = m4;
      }
    }
    if (direct_idx_.empty())
      return;
    for (int li : direct_idx_) {
      GridLevel &L = levels_[li];
      for (int64_t c = 0; c < L.total_cells; ++c)
        L.mu[c] = L.counts[c] > 0 ? L.mu[c] / (double)L.counts[c] : 0.0;
      std::fill(L.m2.begin(), L.m2.end(), 0.0);
      std::fill(L.m3.begin(), L.m3.end(), 0.0);
      std::fill(L.m4.begin(), L.m4.end(), 0.0);
    }
    // Pass 2: central moments for direct levels, single sweep.
    for (int64_t k = 0; k < nv; ++k) {
      const int64_t i = Strided ? sampled_[k] : k;
      const double v = (double)data[i];
      for (int li : direct_idx_) {
        GridLevel &L = levels_[li];
        const int16_t c = L.cell_of[i];
        const double d = v - L.mu[c], d2 = d * d;
        L.m2[c] += d2;
        L.m3[c] += d2 * d;
        L.m4[c] += d2 * d2;
      }
    }
    for (int li : direct_idx_) {
      GridLevel &L = levels_[li];
      for (int64_t c = 0; c < L.total_cells; ++c)
        pack_cell(L, c, nm);
    }
  }

  // float64 fused sweep — every level is direct two-pass (no histogram path).
  template <bool Strided>
  void _run_f64(const double *TS_RESTRICT data, int64_t nv) {
    const int nm = n_moments_;
    for (GridLevel &L : levels_) {
      std::fill(L.mu.begin(), L.mu.end(), 0.0);
      std::fill(L.counts.begin(), L.counts.end(), 0);
    }
    for (int64_t k = 0; k < nv; ++k) {
      const int64_t i = Strided ? sampled_[k] : k;
      const double v = data[i];
      for (GridLevel &L : levels_) {
        const int16_t c = L.cell_of[i];
        L.mu[c] += v;
        L.counts[c]++;
      }
    }
    for (GridLevel &L : levels_) {
      for (int64_t c = 0; c < L.total_cells; ++c)
        L.mu[c] = L.counts[c] > 0 ? L.mu[c] / (double)L.counts[c] : 0.0;
      std::fill(L.m2.begin(), L.m2.end(), 0.0);
      std::fill(L.m3.begin(), L.m3.end(), 0.0);
      std::fill(L.m4.begin(), L.m4.end(), 0.0);
    }
    for (int64_t k = 0; k < nv; ++k) {
      const int64_t i = Strided ? sampled_[k] : k;
      const double v = data[i];
      for (GridLevel &L : levels_) {
        const int16_t c = L.cell_of[i];
        const double d = v - L.mu[c], d2 = d * d;
        L.m2[c] += d2;
        L.m3[c] += d2 * d;
        L.m4[c] += d2 * d2;
      }
    }
    for (GridLevel &L : levels_)
      for (int64_t c = 0; c < L.total_cells; ++c)
        pack_cell(L, c, nm);
  }

public:
  _GridStatsComputerImpl() = default;

  // grids:  list of per-axis cell-exponent vectors (K levels, K >= 1).
  // stride: per-axis stride (all 1 => contiguous full scan).
  void set_config(const std::vector<int64_t> &shape,
                  const std::vector<std::vector<int>> &grids,
                  const std::vector<int64_t> &stride, int n_moments = 4) {
    n_moments_ = n_moments;
    total_ = std::accumulate(shape.begin(), shape.end(), (int64_t)1,
                             std::multiplies<int64_t>{});
    strided_ = std::any_of(stride.begin(), stride.end(),
                           [](int64_t s) { return s > 1; });
    sampled_.clear();
    if (strided_)
      sampled_ = sampled_indices(shape, stride);

    levels_.clear();
    hist_idx_.clear();
    direct_idx_.clear();
    levels_.reserve(grids.size());
    for (const auto &g : grids) {
      levels_.push_back(_make_level(shape, g));
      if (levels_.back().use_hist)
        hist_idx_.push_back((int)levels_.size() - 1);
      else
        direct_idx_.push_back((int)levels_.size() - 1);
    }
  }

  // Returns a list of VIEWS (one per level) into retained out_bufs — the caller
  // copies each before the next compute().
  nb::list compute_u8(const uint8_t *TS_RESTRICT data) {
    const int64_t nv = strided_ ? (int64_t)sampled_.size() : total_;
    if (strided_)
      _run_u8<true>(data, nv);
    else
      _run_u8<false>(data, nv);
    return _views();
  }

  nb::list compute_f64(const double *TS_RESTRICT data) {
    const int64_t nv = strided_ ? (int64_t)sampled_.size() : total_;
    if (strided_)
      _run_f64<true>(data, nv);
    else
      _run_f64<false>(data, nv);
    return _views();
  }

  int64_t n_levels() const { return (int64_t)levels_.size(); }
  int64_t n_moments() const { return n_moments_; }
};

// ---------------------------------------------------------------------------
// Public C++ bindings
// ---------------------------------------------------------------------------
using ArrF64 = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
using ArrF32 = nb::ndarray<nb::numpy, float, nb::c_contig, nb::device::cpu>;
using ArrU8 = nb::ndarray<nb::numpy, uint8_t, nb::c_contig, nb::device::cpu>;

#define MAKE_ENTRY(name, T, ArrT)                                              \
  nb::dict name(ArrT arr, std::vector<std::vector<int>> axes,                  \
                std::vector<int64_t> stride, int n_moments) {                  \
    const int ndim = (int)arr.ndim();                                          \
    std::vector<int64_t> shape(ndim);                                          \
    for (int d = 0; d < ndim; ++d)                                             \
      shape[d] = arr.shape(d);                                                 \
    if ((int)stride.size() != ndim)                                            \
      throw std::invalid_argument("stride length must match ndim");            \
    return compute_typed<T>(arr.data(), shape, axes, stride, n_moments);       \
  }

MAKE_ENTRY(compute_f64, double, ArrF64)
MAKE_ENTRY(compute_f32, float, ArrF32)
MAKE_ENTRY(compute_u8, uint8_t, ArrU8)

NB_MODULE(tensorstats_core, m) {
  m.doc() = "tensorstats internal C++ module. Public API: ts.StatsComputer.";

  m.def("compute_f64", &compute_f64, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));
  m.def("compute_f32", &compute_f32, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));
  m.def("compute_u8", &compute_u8, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));

  // Internal stateful grid computer — not part of the public Python API.
  nb::class_<_GridStatsComputerImpl>(m, "_GridStatsComputerImpl")
      .def(nb::init<>())
      .def("set_config", &_GridStatsComputerImpl::set_config, nb::arg("shape"),
           nb::arg("grids"), nb::arg("stride"), nb::arg("n_moments") = 4)
      .def(
          "compute_u8",
          [](_GridStatsComputerImpl &self, ArrU8 arr) {
            return self.compute_u8(arr.data());
          },
          nb::arg("arr"))
      .def(
          "compute_f64",
          [](_GridStatsComputerImpl &self, ArrF64 arr) {
            return self.compute_f64(arr.data());
          },
          nb::arg("arr"))
      .def_prop_ro("n_levels", &_GridStatsComputerImpl::n_levels)
      .def_prop_ro("n_moments", &_GridStatsComputerImpl::n_moments);
}
