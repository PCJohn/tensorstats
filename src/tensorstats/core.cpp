#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
//   global reduction → (n_moments,)
//   per-channel      → (C, n_moments)
//   grid             → (*cell_shape, n_moments)
//
// Moments layout: [0]=mean  [1]=variance  [2]=m3  [3]=m4
//   Callers derive:  std = sqrt(m[1])
//                    skewness = m[2] / std^3
//                    kurtosis = m[3] / m[1]^2
//
// Computation paths (selected at runtime):
//   Global    — straight loop over all elements, AVX2-friendly
//   LastAxis  — stride-C column loops, no modulo
//   General   — iterative (flat_idx, bucket) enumeration for arbitrary axes
//   Grid      — pixel-outer scatter into per-cell accumulators;
//               precomputed int16 cell_of[] eliminates per-pixel arithmetic
//
// uint8 histogram path (Global + LastAxis):
//   N integer increments + 256 float FMAs per reduction.
//   4-way parallel counters reduce write-after-write stalls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// uint8 histogram helpers
// ---------------------------------------------------------------------------

static void build_hist4(const uint8_t *TS_RESTRICT d, int64_t n,
                        int64_t *TS_RESTRICT hist) {
  int64_t h0[256] = {}, h1[256] = {}, h2[256] = {}, h3[256] = {};
  int64_t i = 0, n4 = (n >> 2) << 2;
  for (; i < n4; i += 4) {
    h0[d[i]]++;
    h1[d[i + 1]]++;
    h2[d[i + 2]]++;
    h3[d[i + 3]]++;
  }
  for (; i < n; ++i)
    h0[d[i]]++;
  for (int v = 0; v < 256; ++v)
    hist[v] = h0[v] + h1[v] + h2[v] + h3[v];
}

static void moments_from_hist(const int64_t *TS_RESTRICT hist, int64_t n,
                              double &mu, double &m2, double &m3, double &m4) {
  int64_t s1 = 0;
  for (int v = 0; v < 256; ++v)
    s1 += hist[v] * (int64_t)v;
  mu = (double)s1 / (double)n;
  m2 = m3 = m4 = 0.0;
  for (int v = 0; v < 256; ++v) {
    if (!hist[v])
      continue;
    double x = (double)v - mu, x2 = x * x, h = (double)hist[v];
    m2 += h * x2;
    m3 += h * x2 * x;
    m4 += h * x2 * x2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
}

static void global_u8_hist(const uint8_t *TS_RESTRICT d, int64_t n,
                           int64_t step, double &mu, double &m2, double &m3,
                           double &m4) {
  int64_t hist[256] = {};
  if (step == 1) {
    build_hist4(d, n, hist);
    moments_from_hist(hist, n, mu, m2, m3, m4);
  } else {
    int64_t h0[256] = {}, h1[256] = {}, h2[256] = {}, h3[256] = {};
    int64_t step4 = step * 4, ns = 0, i = 0, n4 = (n / step4) * step4;
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
    for (int v = 0; v < 256; ++v)
      hist[v] = h0[v] + h1[v] + h2[v] + h3[v];
    moments_from_hist(hist, ns, mu, m2, m3, m4);
  }
}

static void last_axis_u8_hist(const uint8_t *TS_RESTRICT d, int64_t HW,
                              int64_t C, int64_t sr, int64_t sc,
                              std::vector<double> &mu, std::vector<double> &m2,
                              std::vector<double> &m3,
                              std::vector<double> &m4) {
  for (int64_t c = 0; c < C; c += sc) {
    int64_t hist[256] = {}, ns = 0;
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
  int ndim = (int)shape.size();
  int64_t cap = 1;
  for (int d = 0; d < ndim; ++d)
    cap *= (shape[d] + stride[d] - 1) / stride[d];
  std::vector<int64_t> result;
  result.reserve(cap);

  std::vector<int64_t> fs(ndim, 1);
  for (int d = ndim - 2; d >= 0; --d)
    fs[d] = fs[d + 1] * shape[d + 1];

  std::vector<int64_t> coords(ndim, 0);
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
  int64_t ns = (n + step - 1) / step;
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
    double d = (double)data[i] - mu, d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  double inv = 1.0 / (double)ns;
  m2 *= inv;
  m3 *= inv;
  m4 *= inv;
}

template <typename T>
static void global_pass_idx(const T *TS_RESTRICT data,
                            const std::vector<int64_t> &idx, double &mu,
                            double &m2, double &m3, double &m4) {
  int64_t ns = (int64_t)idx.size();
  double s = 0;
  for (int64_t i : idx)
    s += (double)data[i];
  mu = s / (double)ns;
  m2 = m3 = m4 = 0;
  for (int64_t i : idx) {
    double d = (double)data[i] - mu, d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  double inv = 1.0 / (double)ns;
  m2 *= inv;
  m3 *= inv;
  m4 *= inv;
}

template <typename T>
static void last_axis_pass(const T *TS_RESTRICT data, int64_t HW, int64_t C,
                           int64_t sr, int64_t sc, std::vector<double> &mu,
                           std::vector<double> &m2, std::vector<double> &m3,
                           std::vector<double> &m4) {
  int64_t nrows = (HW + sr - 1) / sr;
  double inv = 1.0 / (double)nrows;
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
      double d = (double)data[r * C + c] - mu[c], d2 = d * d;
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
  for (auto [fi, b] : pairs) {
    mu[b] += (double)data[fi];
    counts[b]++;
  }
  for (int64_t b = 0; b < n_buckets; ++b)
    mu[b] = counts[b] > 0 ? mu[b] / (double)counts[b] : 0.0;
  std::fill(m2.begin(), m2.end(), 0.0);
  std::fill(m3.begin(), m3.end(), 0.0);
  std::fill(m4.begin(), m4.end(), 0.0);
  for (auto [fi, b] : pairs) {
    double d = (double)data[fi] - mu[b], d2 = d * d;
    m2[b] += d2;
    m3[b] += d2 * d;
    m4[b] += d2 * d2;
  }
  for (int64_t b = 0; b < n_buckets; ++b)
    if (counts[b] > 0) {
      double inv = 1.0 / (double)counts[b];
      m2[b] *= inv;
      m3[b] *= inv;
      m4[b] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Axis spec
// ---------------------------------------------------------------------------
struct AxisSpec {
  std::vector<int> reduce_axes;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> out_strides;
  int64_t acc_size = 1;
  bool is_last_dim = false;
  int64_t last_dim_C = 1;
};

static AxisSpec make_axis_spec(const std::vector<int> &axes,
                               const std::vector<int64_t> &shape) {
  int ndim = (int)shape.size();
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
    bool red = std::find(s.reduce_axes.begin(), s.reduce_axes.end(), d) !=
               s.reduce_axes.end();
    if (!red) {
      s.out_shape.push_back(shape[d]);
      s.acc_size *= shape[d];
    }
  }
  s.out_strides.resize(s.out_shape.size(), 1);
  for (int i = (int)s.out_shape.size() - 2; i >= 0; --i)
    s.out_strides[i] = s.out_strides[i + 1] * s.out_shape[i + 1];
  if ((int)s.reduce_axes.size() == ndim - 1) {
    bool abl = true;
    for (int i = 0; i < ndim - 1; ++i)
      if (s.reduce_axes[i] != i) {
        abl = false;
        break;
      }
    if (abl) {
      s.is_last_dim = true;
      s.last_dim_C = shape[ndim - 1];
    }
  }
  return s;
}

static std::vector<std::pair<int64_t, int64_t>>
make_pairs(const std::vector<int64_t> &shape,
           const std::vector<int64_t> &stride,
           const std::vector<int> &reduce_axes,
           const std::vector<int64_t> &out_strides) {
  int ndim = (int)shape.size();
  std::vector<bool> is_red(ndim, false);
  std::vector<int64_t> odim(ndim, -1);
  std::vector<int64_t> fs(ndim, 1);
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
// Pack — write accumulated moments into a new heap-allocated ndarray
// ---------------------------------------------------------------------------
static nb::ndarray<nb::numpy, double>
pack(const std::vector<double> &mu, const std::vector<double> &m2,
     const std::vector<double> &m3, const std::vector<double> &m4,
     const std::vector<int64_t> &out_shape, int64_t nacc, int n_moments) {
  std::vector<size_t> sh;
  for (auto d : out_shape)
    sh.push_back((size_t)d);
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
// Axes reduction entry point
// ---------------------------------------------------------------------------
template <typename T>
static nb::dict compute_typed(const T *data, const std::vector<int64_t> &shape,
                              const std::vector<std::vector<int>> &axes_list,
                              const std::vector<int64_t> &stride,
                              int n_moments) {
  int ndim = (int)shape.size();
  int64_t total = 1;
  for (auto d : shape)
    total *= d;

  bool uniform_stride = true;
  int64_t s0 = stride[0];
  for (int d = 1; d < ndim; ++d)
    if (stride[d] != s0) {
      uniform_stride = false;
      break;
    }
  bool has_stride = false;
  for (int d = 0; d < ndim; ++d)
    if (stride[d] > 1) {
      has_stride = true;
      break;
    }

  nb::dict result;

  for (auto &raw_axes : axes_list) {
    bool is_global = raw_axes.empty();

    if (is_global) {
      double mu = 0, m2 = 0, m3 = 0, m4 = 0;
      if constexpr (std::is_same_v<T, uint8_t>) {
        if (!has_stride || uniform_stride)
          global_u8_hist(data, total, uniform_stride ? s0 : 1, mu, m2, m3, m4);
        else {
          auto idx = sampled_indices(shape, stride);
          global_pass_idx<T>(data, idx, mu, m2, m3, m4);
        }
      } else {
        if (!has_stride)
          global_pass<T>(data, total, 1, mu, m2, m3, m4);
        else if (uniform_stride)
          global_pass<T>(data, total, s0, mu, m2, m3, m4);
        else {
          auto idx = sampled_indices(shape, stride);
          global_pass_idx<T>(data, idx, mu, m2, m3, m4);
        }
      }
      std::vector<size_t> sh = {(size_t)n_moments};
      auto *p = new double[n_moments];
      if (n_moments >= 1)
        p[0] = mu;
      if (n_moments >= 2)
        p[1] = m2;
      if (n_moments >= 3)
        p[2] = m3;
      if (n_moments >= 4)
        p[3] = m4;
      nb::capsule own(
          p, [](void *x) noexcept { delete[] static_cast<double *>(x); });
      result["global"] = nb::ndarray<nb::numpy, double>(p, 1, sh.data(), own);

    } else {
      AxisSpec spec = make_axis_spec(raw_axes, shape);
      int64_t nacc = spec.acc_size;
      std::vector<double> mu(nacc), m2(nacc), m3(nacc), m4(nacc);

      int ndim_red = (int)spec.reduce_axes.size();
      bool simple_hw_stride = true;
      if (ndim_red > 1 && has_stride) {
        int64_t sr_check = stride[spec.reduce_axes[0]];
        for (int i = 1; i < ndim_red; ++i)
          if (stride[spec.reduce_axes[i]] != sr_check) {
            simple_hw_stride = false;
            break;
          }
        if (simple_hw_stride && ndim_red > 1)
          for (int i = 1; i < ndim_red; ++i)
            if (stride[spec.reduce_axes[i]] != 1) {
              simple_hw_stride = false;
              break;
            }
      }

      if (spec.is_last_dim) {
        int64_t HW = total / spec.last_dim_C;
        if constexpr (std::is_same_v<T, uint8_t>) {
          if (!has_stride)
            last_axis_u8_hist(data, HW, spec.last_dim_C, 1, 1, mu, m2, m3, m4);
          else if (simple_hw_stride && ndim_red == 1)
            last_axis_u8_hist(data, HW, spec.last_dim_C,
                              stride[spec.reduce_axes[0]], stride[ndim - 1], mu,
                              m2, m3, m4);
          else {
            auto p =
                make_pairs(shape, stride, spec.reduce_axes, spec.out_strides);
            general_pass<T>(data, p, nacc, mu, m2, m3, m4);
          }
        } else if (!has_stride)
          last_axis_pass<T>(data, HW, spec.last_dim_C, 1, 1, mu, m2, m3, m4);
        else if (simple_hw_stride && ndim_red == 1)
          last_axis_pass<T>(data, HW, spec.last_dim_C,
                            stride[spec.reduce_axes[0]], stride[ndim - 1], mu,
                            m2, m3, m4);
        else {
          auto p =
              make_pairs(shape, stride, spec.reduce_axes, spec.out_strides);
          general_pass<T>(data, p, nacc, mu, m2, m3, m4);
        }
      } else {
        auto p = make_pairs(shape, stride, spec.reduce_axes, spec.out_strides);
        general_pass<T>(data, p, nacc, mu, m2, m3, m4);
      }

      std::string key;
      for (int i = 0; i < (int)raw_axes.size(); ++i) {
        if (i)
          key += ",";
        key += std::to_string(raw_axes[i]);
      }
      result[key.c_str()] =
          pack(mu, m2, m3, m4, spec.out_shape, nacc, n_moments);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// _GridStatsComputerImpl — internal stateful grid moment computer.
//
// Not exposed publicly. Used by StatsComputer (Python) via nanobind.
//
// Computes exact central moments [mean, variance, m3, m4] for each grid cell.
// Callers derive std/skewness/kurtosis from these raw moments.
//
// Retained across calls (for fixed shape + grid config):
//   cell_of_[]   int16 flat cell index per pixel. Precomputed at construction.
//                Eliminates per-pixel index arithmetic in scatter loops.
//                Range [0, total_cells-1] fits int16 (total_cells <= 32767).
//   sums_, counts_, mu_, m2_, m3_, m4_   Accumulator vectors (total_cells).
//                Zero-filled each call instead of reallocated.
//   out_buf_[]   Output buffer (total_cells * n_moments doubles).
//                compute_u8/compute_f64 return a VIEW into this buffer with
//                nb::none() as owner — zero allocation per call.
//                Caller must copy if the array must survive the next call.
//
// Grid cell assignment:
//   cell_of[pixel] = sum_d( (coord[d] * n_cells[d] / shape[d]) * cell_stride[d]
//   ) Integer floor-division gives uniform cell boundaries for any shape.
//
// uint8 histogram path (pixels_per_cell >= HIST_THRESHOLD):
//   Per-cell hist[256] + 256 FMA finalise. Faster for large cells.
//
// Direct two-pass (small cells, float types):
//   Pass 1: scatter pixels → per-cell sums. Derive means.
//   Pass 2: scatter (pixel - mean)^k → per-cell moment accumulators.
//   #pragma loop(ivdep) hints MSVC that scatter writes are independent
//   (different pixels always map to the same cell — no loop-carried dependency
//   on the accumulator write side that MSVC can prove statically).
// ---------------------------------------------------------------------------
static constexpr int64_t HIST_THRESHOLD = 256;

class _GridStatsComputerImpl {
  std::vector<int64_t> shape_, grid_;
  std::vector<size_t> out_shape_;
  int64_t total_ = 0;
  int64_t total_cells_ = 0;
  int64_t n_moments_ = 4;
  bool use_hist_ = false;

  std::vector<int16_t> cell_of_;
  std::vector<double> sums_, mu_, m2_, m3_, m4_, out_buf_;
  std::vector<int64_t> counts_, hists_;

  void _rebuild(const std::vector<int64_t> &shape, const std::vector<int> &grid,
                int n_moments) {
    shape_ = shape;
    grid_ = std::vector<int64_t>(grid.begin(), grid.end());
    n_moments_ = n_moments;
    int ndim = (int)shape.size();

    std::vector<int64_t> n_cells(ndim), cs(ndim, 1);
    for (int d = 0; d < ndim; ++d)
      n_cells[d] = (int64_t)1 << grid[d];
    total_cells_ = 1;
    for (int d = 0; d < ndim; ++d)
      total_cells_ *= n_cells[d];
    for (int d = ndim - 2; d >= 0; --d)
      cs[d] = cs[d + 1] * n_cells[d + 1];

    total_ = 1;
    for (auto s : shape)
      total_ *= s;
    int64_t px_per_cell = (total_cells_ > 0) ? (total_ / total_cells_) : 0;
    use_hist_ = (px_per_cell >= HIST_THRESHOLD);

    if (total_cells_ > 32767)
      throw std::runtime_error(
          "_GridStatsComputerImpl: total_cells exceeds int16 range");

    // Precompute per-axis cell luts then flatten to cell_of_
    std::vector<std::vector<int64_t>> lut(ndim);
    for (int d = 0; d < ndim; ++d) {
      lut[d].resize(shape[d]);
      for (int64_t i = 0; i < shape[d]; ++i)
        lut[d][i] = i * n_cells[d] / shape[d];
    }
    cell_of_.resize(total_);
    std::vector<int64_t> coords(ndim, 0);
    for (int64_t flat = 0; flat < total_; ++flat) {
      int64_t cell = 0;
      for (int d = 0; d < ndim; ++d)
        cell += lut[d][coords[d]] * cs[d];
      cell_of_[flat] = (int16_t)cell;
      for (int d = ndim - 1; d >= 0; --d) {
        if (++coords[d] < shape[d])
          break;
        coords[d] = 0;
      }
    }

    sums_.assign(total_cells_, 0.0);
    mu_.assign(total_cells_, 0.0);
    m2_.assign(total_cells_, 0.0);
    m3_.assign(total_cells_, 0.0);
    m4_.assign(total_cells_, 0.0);
    counts_.assign(total_cells_, 0);
    if (use_hist_)
      hists_.assign(total_cells_ * 256, 0);
    out_buf_.resize(total_cells_ * n_moments_);

    out_shape_.clear();
    for (int d = 0; d < ndim; ++d)
      out_shape_.push_back((size_t)n_cells[d]);
    out_shape_.push_back((size_t)n_moments_);
  }

  nb::ndarray<nb::numpy, double> _view() {
    return nb::ndarray<nb::numpy, double>(out_buf_.data(), out_shape_.size(),
                                          out_shape_.data(), nb::none());
  }

  void _pack(int64_t c) {
    if (!counts_[c])
      return;
    double inv = 1.0 / (double)counts_[c];
    double *out = out_buf_.data() + c * n_moments_;
    if (n_moments_ >= 1)
      out[0] = mu_[c];
    if (n_moments_ >= 2)
      out[1] = m2_[c] * inv;
    if (n_moments_ >= 3)
      out[2] = m3_[c] * inv;
    if (n_moments_ >= 4)
      out[3] = m4_[c] * inv;
  }

public:
  _GridStatsComputerImpl() = default;

  void set_config(const std::vector<int64_t> &shape,
                  const std::vector<int> &grid, int n_moments = 4) {
    _rebuild(shape, grid, n_moments);
  }

  // compute_u8: exact moments over all pixels in data.
  // Returns VIEW into retained out_buf_ — copy before the next call.
  nb::ndarray<nb::numpy, double> compute_u8(const uint8_t *TS_RESTRICT data) {
    if (use_hist_) {
      std::fill(hists_.begin(), hists_.end(), 0);
      std::fill(counts_.begin(), counts_.end(), 0);
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
      for (int64_t i = 0; i < total_; ++i) {
        int16_t cell = cell_of_[i];
        hists_[cell * 256 + data[i]]++;
        counts_[cell]++;
      }
      for (int64_t cell = 0; cell < total_cells_; ++cell) {
        if (!counts_[cell])
          continue;
        double mu, m2, m3, m4;
        moments_from_hist(hists_.data() + cell * 256, counts_[cell], mu, m2, m3,
                          m4);
        double *out = out_buf_.data() + cell * n_moments_;
        if (n_moments_ >= 1)
          out[0] = mu;
        if (n_moments_ >= 2)
          out[1] = m2;
        if (n_moments_ >= 3)
          out[2] = m3;
        if (n_moments_ >= 4)
          out[3] = m4;
      }
    } else {
      std::fill(sums_.begin(), sums_.end(), 0.0);
      std::fill(counts_.begin(), counts_.end(), 0);
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
      for (int64_t i = 0; i < total_; ++i) {
        sums_[cell_of_[i]] += (double)data[i];
        counts_[cell_of_[i]]++;
      }
      for (int64_t c = 0; c < total_cells_; ++c)
        mu_[c] = counts_[c] > 0 ? sums_[c] / (double)counts_[c] : 0.0;
      std::fill(m2_.begin(), m2_.end(), 0.0);
      std::fill(m3_.begin(), m3_.end(), 0.0);
      std::fill(m4_.begin(), m4_.end(), 0.0);
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
      for (int64_t i = 0; i < total_; ++i) {
        int16_t cell = cell_of_[i];
        double d = (double)data[i] - mu_[cell], d2 = d * d;
        m2_[cell] += d2;
        m3_[cell] += d2 * d;
        m4_[cell] += d2 * d2;
      }
      for (int64_t c = 0; c < total_cells_; ++c)
        _pack(c);
    }
    return _view();
  }

  // compute_f64: exact moments over all pixels (float64 input).
  nb::ndarray<nb::numpy, double> compute_f64(const double *TS_RESTRICT data) {
    std::fill(sums_.begin(), sums_.end(), 0.0);
    std::fill(counts_.begin(), counts_.end(), 0);
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (int64_t i = 0; i < total_; ++i) {
      sums_[cell_of_[i]] += data[i];
      counts_[cell_of_[i]]++;
    }
    for (int64_t c = 0; c < total_cells_; ++c)
      mu_[c] = counts_[c] > 0 ? sums_[c] / (double)counts_[c] : 0.0;
    std::fill(m2_.begin(), m2_.end(), 0.0);
    std::fill(m3_.begin(), m3_.end(), 0.0);
    std::fill(m4_.begin(), m4_.end(), 0.0);
#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
    for (int64_t i = 0; i < total_; ++i) {
      int16_t cell = cell_of_[i];
      double d = data[i] - mu_[cell], d2 = d * d;
      m2_[cell] += d2;
      m3_[cell] += d2 * d;
      m4_[cell] += d2 * d2;
    }
    for (int64_t c = 0; c < total_cells_; ++c)
      _pack(c);
    return _view();
  }

  int64_t total_cells() const { return total_cells_; }
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
    int ndim = (int)arr.ndim();                                                \
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
  m.doc() = "tensorstats internal C++ module. Use the Python API: "
            "ts.StatsComputer, ts.compute.";

  m.def("compute_f64", &compute_f64, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));
  m.def("compute_f32", &compute_f32, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));
  m.def("compute_u8", &compute_u8, nb::arg("arr"), nb::arg("axes"),
        nb::arg("stride"), nb::arg("n_moments"));

  // Internal stateful grid computer. Not part of the public API.
  nb::class_<_GridStatsComputerImpl>(m, "_GridStatsComputerImpl")
      .def(nb::init<>())
      .def("set_config", &_GridStatsComputerImpl::set_config, nb::arg("shape"),
           nb::arg("grid"), nb::arg("n_moments") = 4)
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
      .def_prop_ro("total_cells", &_GridStatsComputerImpl::total_cells)
      .def_prop_ro("n_moments", &_GridStatsComputerImpl::n_moments);
}
