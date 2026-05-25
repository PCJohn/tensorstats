#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <string>
#include <cstdint>


// Portable no-alias hint: __restrict__ on GCC/Clang, __restrict on MSVC
#if defined(_MSC_VER)
#  define TS_RESTRICT __restrict
#else
#  define TS_RESTRICT __restrict__
#endif

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// tensorstats — fast two-pass central moment computation
//
// Algorithm: two-pass, numerically stable:
//   Pass 1: mean per bucket  — pure sum, SIMD-vectorized
//   Pass 2: d=x-mu, accumulate d^2, d^2*d, d^2*d^2 — 4 FMAs/element
//
// Specialised inner-loop paths (selected at runtime):
//   Global   — single accumulator, straight loop, fastest
//   LastAxis — stride-C column loops (no modulo) — e.g. per-channel HxWxC
//   General  — arbitrary axis combinations via precomputed index array
//
// stride parameter: subsample the array without malloc/resize.
//   For the Global path, stride is applied as a flat step: data[0], data[s],
//   data[2s], ... This is equivalent to arr[::s0, ::s1, ...].ravel() only
//   when the strides are uniform; for correctness across axis combinations we
//   use a precomputed list of sampled flat indices.
//
// Performance notes:
//   • float32 input: 2× faster than float64 on AVX2 (8 floats vs 4 doubles)
//   • uint8 global/last-axis: histogram path — builds hist[256] in one pass,
//     then computes moments from 256 bins rather than N elements. 1.3–1.6×
//     faster than the direct loop because 256 float FMAs replace N float FMAs.
//     Uses 4-way split (h0/h1/h2/h3) to reduce write-after-write stalls.
//     Numerically stable: mean computed from exact integer sum; central moments
//     computed as Σ hist[v]*(v-mu)^k directly (no raw-moment subtraction).
//   • uint8 general (arbitrary axes): direct loop (histogram doesn't simplify).
//   • stride=2 on global path: ~2× faster (half elements, same loop structure)
//   • Strided last-axis path: column loops with sr/sc strides, still vectorized
//   • TS_RESTRICT on all data pointers: no-alias hint for the compiler
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// uint8 histogram helpers
// ---------------------------------------------------------------------------

// Build a 256-bin histogram using 4-way parallel counters to reduce
// write-after-write stalls on the same bin (improves out-of-order scheduling).
static void build_hist4(const uint8_t* TS_RESTRICT d, int64_t n,
                         int64_t* TS_RESTRICT hist)
{
    int64_t h0[256]={}, h1[256]={}, h2[256]={}, h3[256]={};
    int64_t i=0, n4=(n>>2)<<2;
    for (; i < n4; i += 4) {
        h0[d[i  ]]++;
        h1[d[i+1]]++;
        h2[d[i+2]]++;
        h3[d[i+3]]++;
    }
    for (; i < n; ++i) h0[d[i]]++;
    for (int v = 0; v < 256; ++v) hist[v] = h0[v]+h1[v]+h2[v]+h3[v];
}

// Two-pass central moments from a 256-bin histogram.
// Pass 1 (mean): exact integer sum Σ hist[v]*v — no float rounding.
// Pass 2 (moments): only 256 float iterations regardless of N.
static void moments_from_hist(const int64_t* TS_RESTRICT hist, int64_t n,
                               double &mu, double &m2, double &m3, double &m4)
{
    int64_t s1 = 0;
    for (int v = 0; v < 256; ++v) s1 += hist[v] * (int64_t)v;
    mu = (double)s1 / (double)n;
    m2 = m3 = m4 = 0.0;
    for (int v = 0; v < 256; ++v) {
        if (!hist[v]) continue;
        double x = (double)v - mu, x2 = x*x, h = (double)hist[v];
        m2 += h*x2; m3 += h*x2*x; m4 += h*x2*x2;
    }
    m2 /= n; m3 /= n; m4 /= n;
}

// Global uint8: histogram + moments. stride > 1 samples every stride-th element.
static void global_u8_hist(const uint8_t* TS_RESTRICT d, int64_t n,
                            int64_t step,
                            double &mu, double &m2, double &m3, double &m4)
{
    if (step == 1) {
        int64_t hist[256] = {};
        build_hist4(d, n, hist);
        moments_from_hist(hist, n, mu, m2, m3, m4);
    } else {
        // Strided: collect sampled values into histogram
        int64_t hist[256] = {};
        int64_t ns = 0;
        for (int64_t i = 0; i < n; i += step) { hist[d[i]]++; ns++; }
        moments_from_hist(hist, ns, mu, m2, m3, m4);
    }
}

// Per-channel uint8: one histogram per channel, strided column access.
// Layout: data[r*C + c], r in [0, HW), c in [0, C).
static void last_axis_u8_hist(const uint8_t* TS_RESTRICT d,
                               int64_t HW, int64_t C,
                               int64_t sr, int64_t sc,
                               std::vector<double> &mu,
                               std::vector<double> &m2,
                               std::vector<double> &m3,
                               std::vector<double> &m4)
{
    for (int64_t c = 0; c < C; c += sc) {
        int64_t hist[256] = {};
        int64_t ns = 0;
        for (int64_t r = 0; r < HW; r += sr) { hist[d[r*C + c]]++; ns++; }
        moments_from_hist(hist, ns, mu[c], m2[c], m3[c], m4[c]);
    }
}

// ---------------------------------------------------------------------------
// Build flat indices of sampled elements given per-axis strides
// ---------------------------------------------------------------------------
static std::vector<int64_t> sampled_indices(
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &stride)
{
    int ndim = (int)shape.size();
    std::vector<int64_t> result;
    // Estimate capacity
    int64_t cap = 1;
    for (int d = 0; d < ndim; ++d) cap *= (shape[d] + stride[d] - 1) / stride[d];
    result.reserve(cap);

    // Iterate multi-index with per-axis strides
    std::vector<int64_t> coords(ndim, 0);
    // Precompute flat strides (C-order)
    std::vector<int64_t> flat_stride(ndim, 1);
    for (int d = ndim-2; d >= 0; --d)
        flat_stride[d] = flat_stride[d+1] * shape[d+1];

    std::function<void(int, int64_t)> recurse = [&](int d, int64_t flat) {
        if (d == ndim) { result.push_back(flat); return; }
        for (int64_t c = 0; c < shape[d]; c += stride[d])
            recurse(d+1, flat + c * flat_stride[d]);
    };
    recurse(0, 0);
    return result;
}

// ---------------------------------------------------------------------------
// Global two-pass: straight loop with flat step (= uniform stride)
// ---------------------------------------------------------------------------
template<typename T>
static void global_pass(const T* TS_RESTRICT data,
                         int64_t n, int64_t step,
                         double &out_mu,
                         double &out_m2, double &out_m3, double &out_m4)
{
    int64_t ns = (n + step - 1) / step;
    double s = 0.0;
    for (int64_t i = 0; i < n; i += step) s += (double)data[i];
    double mu = s / (double)ns;
    double m2=0, m3=0, m4=0;
    for (int64_t i = 0; i < n; i += step) {
        double d = (double)data[i] - mu, d2 = d*d;
        m2 += d2; m3 += d2*d; m4 += d2*d2;
    }
    double inv = 1.0 / (double)ns;
    out_mu=mu; out_m2=m2*inv; out_m3=m3*inv; out_m4=m4*inv;
}

// Global two-pass via precomputed sampled index list (non-uniform strides)
template<typename T>
static void global_pass_idx(const T* TS_RESTRICT data,
                              const std::vector<int64_t> &idx,
                              double &out_mu,
                              double &out_m2, double &out_m3, double &out_m4)
{
    int64_t ns = (int64_t)idx.size();
    double s = 0.0;
    for (int64_t i : idx) s += (double)data[i];
    double mu = s / (double)ns;
    double m2=0, m3=0, m4=0;
    for (int64_t i : idx) {
        double d = (double)data[i] - mu, d2 = d*d;
        m2 += d2; m3 += d2*d; m4 += d2*d2;
    }
    double inv = 1.0 / (double)ns;
    out_mu=mu; out_m2=m2*inv; out_m3=m3*inv; out_m4=m4*inv;
}

// ---------------------------------------------------------------------------
// LastAxis two-pass: stride-C column loops, optional row/col strides
// Input layout: data[r*C + c], r in [0, HW), c in [0, C)
// sr: stride over the HW dimension (rows of the channel-interleaved array)
// sc: stride over the C dimension (which channels to sample)
// ---------------------------------------------------------------------------
template<typename T>
static void last_axis_pass(const T* TS_RESTRICT data,
                            int64_t HW, int64_t C,
                            int64_t sr, int64_t sc,
                            std::vector<double> &mu,
                            std::vector<double> &m2,
                            std::vector<double> &m3,
                            std::vector<double> &m4)
{
    int64_t nrows = (HW + sr - 1) / sr;
    double inv = 1.0 / (double)nrows;

    for (int64_t c = 0; c < C; c += sc) {
        double s = 0.0;
        for (int64_t r = 0; r < HW; r += sr) s += (double)data[r*C + c];
        mu[c] = s * inv;
        double s2=0, s3=0, s4=0;
        for (int64_t r = 0; r < HW; r += sr) {
            double d = (double)data[r*C + c] - mu[c], d2 = d*d;
            s2 += d2; s3 += d2*d; s4 += d2*d2;
        }
        m2[c]=s2*inv; m3[c]=s3*inv; m4[c]=s4*inv;
    }
}

// ---------------------------------------------------------------------------
// General two-pass: arbitrary axes via precomputed (flat_idx, bucket) pairs
// ---------------------------------------------------------------------------
template<typename T>
static void general_pass(const T* TS_RESTRICT data,
                          const std::vector<std::pair<int64_t,int64_t>> &pairs,
                          int64_t n_buckets,
                          std::vector<double> &mu,
                          std::vector<double> &m2,
                          std::vector<double> &m3,
                          std::vector<double> &m4)
{
    std::fill(mu.begin(), mu.end(), 0.0);
    std::vector<int64_t> counts(n_buckets, 0);
    for (auto [fi, b] : pairs) { mu[b] += (double)data[fi]; counts[b]++; }
    for (int64_t b=0;b<n_buckets;++b)
        mu[b] = counts[b]>0 ? mu[b]/(double)counts[b] : 0.0;
    std::fill(m2.begin(), m2.end(), 0.0);
    std::fill(m3.begin(), m3.end(), 0.0);
    std::fill(m4.begin(), m4.end(), 0.0);
    for (auto [fi, b] : pairs) {
        double d=(double)data[fi]-mu[b], d2=d*d;
        m2[b]+=d2; m3[b]+=d2*d; m4[b]+=d2*d2;
    }
    for (int64_t b=0;b<n_buckets;++b) if(counts[b]>0){
        double inv=1.0/(double)counts[b];
        m2[b]*=inv; m3[b]*=inv; m4[b]*=inv;
    }
}

// ---------------------------------------------------------------------------
// Axis spec helpers
// ---------------------------------------------------------------------------
struct AxisSpec {
    std::vector<int>     reduce_axes;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_strides;
    int64_t              acc_size    = 1;
    bool                 is_last_dim = false;
    int64_t              last_dim_C  = 1;
};

static AxisSpec make_axis_spec(const std::vector<int>     &axes,
                               const std::vector<int64_t> &shape)
{
    int ndim = (int)shape.size();
    AxisSpec s;
    for (int a : axes) {
        int na = (a<0)?a+ndim:a;
        if(na<0||na>=ndim) throw std::out_of_range("axis out of range");
        s.reduce_axes.push_back(na);
    }
    std::sort(s.reduce_axes.begin(), s.reduce_axes.end());
    s.reduce_axes.erase(std::unique(s.reduce_axes.begin(),s.reduce_axes.end()),
                        s.reduce_axes.end());
    s.acc_size=1;
    for(int d=0;d<ndim;++d){
        bool red=std::find(s.reduce_axes.begin(),s.reduce_axes.end(),d)
                 !=s.reduce_axes.end();
        if(!red){s.out_shape.push_back(shape[d]);s.acc_size*=shape[d];}
    }
    s.out_strides.resize(s.out_shape.size(),1);
    for(int i=(int)s.out_shape.size()-2;i>=0;--i)
        s.out_strides[i]=s.out_strides[i+1]*s.out_shape[i+1];
    if((int)s.reduce_axes.size()==ndim-1){
        bool abl=true;
        for(int i=0;i<ndim-1;++i) if(s.reduce_axes[i]!=i){abl=false;break;}
        if(abl){s.is_last_dim=true;s.last_dim_C=shape[ndim-1];}
    }
    return s;
}

// Build (flat_index, bucket_index) pairs for all sampled elements
static std::vector<std::pair<int64_t,int64_t>> make_pairs(
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &stride,
    const std::vector<int>     &reduce_axes,
    const std::vector<int64_t> &out_strides)
{
    int ndim=(int)shape.size();
    std::vector<std::pair<int64_t,int64_t>> pairs;
    // flat stride (C-order)
    std::vector<int64_t> fs(ndim,1);
    for(int d=ndim-2;d>=0;--d) fs[d]=fs[d+1]*shape[d+1];

    std::function<void(int,int64_t,int64_t)> rec=[&](int d,int64_t flat,int64_t bkt){
        if(d==ndim){pairs.push_back({flat,bkt});return;}
        bool red=std::find(reduce_axes.begin(),reduce_axes.end(),d)
                 !=reduce_axes.end();
        int64_t od=0;
        for(int dd2=0;dd2<d;++dd2)
            if(std::find(reduce_axes.begin(),reduce_axes.end(),dd2)
               ==reduce_axes.end()) ++od;
        for(int64_t c=0;c<shape[d];c+=stride[d]){
            int64_t nb2=bkt+(red?0:c*out_strides[od]);
            rec(d+1,flat+c*fs[d],nb2);
        }
    };
    rec(0,0,0);
    return pairs;
}

// ---------------------------------------------------------------------------
// Pack moments into (n_moments, *out_shape) ndarray
// ---------------------------------------------------------------------------
static nb::ndarray<nb::numpy,double> pack(
    const std::vector<double> &mu, const std::vector<double> &m2,
    const std::vector<double> &m3, const std::vector<double> &m4,
    const AxisSpec &spec, int n_moments)
{
    int64_t nacc=spec.acc_size;
    std::vector<size_t> sh;
    sh.push_back((size_t)n_moments);
    for(auto d:spec.out_shape) sh.push_back((size_t)d);
    auto *ptr=new double[(size_t)n_moments*(size_t)nacc];
    if(n_moments>=1) std::memcpy(ptr+0*nacc,mu.data(),nacc*sizeof(double));
    if(n_moments>=2) std::memcpy(ptr+1*nacc,m2.data(),nacc*sizeof(double));
    if(n_moments>=3) std::memcpy(ptr+2*nacc,m3.data(),nacc*sizeof(double));
    if(n_moments>=4) std::memcpy(ptr+3*nacc,m4.data(),nacc*sizeof(double));
    nb::capsule own(ptr,[](void*p)noexcept{delete[]static_cast<double*>(p);});
    return nb::ndarray<nb::numpy,double>(ptr,sh.size(),sh.data(),own);
}

// ---------------------------------------------------------------------------
// Typed entry point
// ---------------------------------------------------------------------------
template<typename T>
static nb::dict compute_typed(
    const T *data,
    const std::vector<int64_t> &shape,
    const std::vector<std::vector<int>> &axes_list,
    const std::vector<int64_t> &stride,
    int n_moments)
{
    int ndim=(int)shape.size();
    int64_t total=1; for(auto d:shape) total*=d;

    bool uniform_stride=true;
    int64_t s0=stride[0];
    for(int d=1;d<ndim;++d) if(stride[d]!=s0){uniform_stride=false;break;}
    bool has_stride=false;
    for(int d=0;d<ndim;++d) if(stride[d]>1){has_stride=true;break;}

    nb::dict result;

    for(auto &raw_axes:axes_list){
        bool is_global=raw_axes.empty();

        if(is_global){
            double mu=0,m2=0,m3=0,m4=0;
            if constexpr (std::is_same_v<T, uint8_t>) {
                // uint8 histogram path: 1.3-1.6x faster than direct loop.
                // Uniform stride: sample every step-th element into histogram.
                // Non-uniform stride: fall through to sampled-index path.
                if (!has_stride || uniform_stride) {
                    global_u8_hist(data, total, uniform_stride ? s0 : 1,
                                   mu, m2, m3, m4);
                } else {
                    auto idx = sampled_indices(shape, stride);
                    global_pass_idx<T>(data, idx, mu, m2, m3, m4);
                }
            } else {
                if(!has_stride){
                    global_pass<T>(data,total,1,mu,m2,m3,m4);
                } else if(uniform_stride){
                    global_pass<T>(data,total,s0,mu,m2,m3,m4);
                } else {
                    auto idx=sampled_indices(shape,stride);
                    global_pass_idx<T>(data,idx,mu,m2,m3,m4);
                }
            }
            std::vector<size_t> sh={(size_t)n_moments};
            auto *ptr=new double[n_moments];
            if(n_moments>=1)ptr[0]=mu; if(n_moments>=2)ptr[1]=m2;
            if(n_moments>=3)ptr[2]=m3; if(n_moments>=4)ptr[3]=m4;
            nb::capsule own(ptr,[](void*p)noexcept{delete[]static_cast<double*>(p);});
            result["global"]=nb::ndarray<nb::numpy,double>(ptr,1,sh.data(),own);

        } else {
            AxisSpec spec=make_axis_spec(raw_axes,shape);
            int64_t nacc=spec.acc_size;
            std::vector<double> mu(nacc),m2(nacc),m3(nacc),m4(nacc);

            if(spec.is_last_dim){
                // LastAxis path: stride over HW rows and C channels
                // HW = product of all non-last dims
                int64_t HW=total/spec.last_dim_C;
                // sr = product of per-axis strides over all non-last dims
                // For a (H,W,C) tensor: HW=H*W, sr iterates with step sr over
                // the flattened HW dimension.
                // Correct sr: we want to sample every sr-th element in HW.
                // If shape=(H,W,C) and stride=(sh,sw,sc):
                //   sample row (h,w) iff h%sh==0 AND w%sw==0
                //   in flat HW: r = h*W+w; we want r s.t. h%sh==0 && w%sw==0
                //   This is NOT a simple flat stride unless sw==1 or W is a multiple.
                // For correctness: fall through to general path when ndim>2 non-last dims.
                // For the common case (H,W,C) with stride=(s,s,1) or all uniform:
                int ndim_red=(int)spec.reduce_axes.size();
                bool simple_hw_stride=true;
                if(ndim_red>1 && has_stride){
                    // Check if all reduced-axis strides are equal and
                    // non-last stride produces a simple flat step
                    int64_t sr_check=stride[spec.reduce_axes[0]];
                    for(int i=1;i<ndim_red;++i)
                        if(stride[spec.reduce_axes[i]]!=sr_check)
                            {simple_hw_stride=false;break;}
                    // Also verify shape allows simple flat stepping:
                    // For (H,W), flattened step = sr when W is divisible by sr
                    // or when only 1 reduced dim. Otherwise use general.
                    if(simple_hw_stride && ndim_red>1){
                        // Check if inner reduced dims have stride 1
                        for(int i=1;i<ndim_red;++i)
                            if(stride[spec.reduce_axes[i]]!=1)
                                {simple_hw_stride=false;break;}
                        // Only outermost reduced dim has stride > 1
                    }
                }

                if constexpr (std::is_same_v<T, uint8_t>) {
                    // uint8 per-channel histogram: one hist[256] per channel.
                    // Falls back to general for complex multi-dim strides.
                    if (!has_stride) {
                        last_axis_u8_hist(data, HW, spec.last_dim_C,
                                          1, 1, mu, m2, m3, m4);
                    } else if (simple_hw_stride && ndim_red == 1) {
                        int64_t sr = stride[spec.reduce_axes[0]];
                        int64_t sc = stride[ndim-1];
                        last_axis_u8_hist(data, HW, spec.last_dim_C,
                                          sr, sc, mu, m2, m3, m4);
                    } else {
                        auto pairs = make_pairs(shape, stride, spec.reduce_axes,
                                                spec.out_strides);
                        general_pass<T>(data, pairs, nacc, mu, m2, m3, m4);
                    }
                } else if(!has_stride){
                    last_axis_pass<T>(data,HW,spec.last_dim_C,1,1,mu,m2,m3,m4);
                } else if(simple_hw_stride && ndim_red==1){
                    // Single reduced dim with stride
                    int64_t sr=stride[spec.reduce_axes[0]];
                    int64_t sc=stride[ndim-1];
                    last_axis_pass<T>(data,HW,spec.last_dim_C,sr,sc,mu,m2,m3,m4);
                } else {
                    // Fall through to general for complex multi-dim stride
                    auto pairs=make_pairs(shape,stride,spec.reduce_axes,spec.out_strides);
                    general_pass<T>(data,pairs,nacc,mu,m2,m3,m4);
                }
            } else {
                auto pairs=make_pairs(shape,stride,spec.reduce_axes,spec.out_strides);
                general_pass<T>(data,pairs,nacc,mu,m2,m3,m4);
            }

            std::string key;
            for(int i=0;i<(int)raw_axes.size();++i){
                if(i)key+=","; key+=std::to_string(raw_axes[i]);
            }
            result[key.c_str()]=pack(mu,m2,m3,m4,spec,n_moments);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public bindings: float64, float32, uint8
// ---------------------------------------------------------------------------
using ArrF64=nb::ndarray<nb::numpy,double, nb::c_contig,nb::device::cpu>;
using ArrF32=nb::ndarray<nb::numpy,float,  nb::c_contig,nb::device::cpu>;
using ArrU8 =nb::ndarray<nb::numpy,uint8_t,nb::c_contig,nb::device::cpu>;

#define MAKE_ENTRY(name, T, ArrT)                                   \
nb::dict name(ArrT arr,                                             \
              std::vector<std::vector<int>> axes,                   \
              std::vector<int64_t> stride, int n_moments){          \
    int ndim=(int)arr.ndim();                                       \
    std::vector<int64_t> shape(ndim);                               \
    for(int d=0;d<ndim;++d) shape[d]=arr.shape(d);                 \
    if((int)stride.size()!=ndim)                                    \
        throw std::invalid_argument("stride length must match ndim");\
    return compute_typed<T>(arr.data(),shape,axes,stride,n_moments);\
}

MAKE_ENTRY(compute_f64, double,  ArrF64)
MAKE_ENTRY(compute_f32, float,   ArrF32)
MAKE_ENTRY(compute_u8,  uint8_t, ArrU8)

NB_MODULE(tensorstats_core, m){
    m.doc()=R"(
Fast two-pass central moment computation.

Specialised paths (selected at runtime):
  global   — straight loop, AVX2-vectorized, ~2x faster per halved element count with stride
  last-dim — stride-C column loops (no modulo), AVX2-vectorized
  general  — precomputed (flat_index, bucket) pairs, arbitrary axes + strides

Accepts float64, float32, uint8 natively.
stride parameter: skip elements without malloc/resize.
    )";
    m.def("compute_f64",&compute_f64,nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"));
    m.def("compute_f32",&compute_f32,nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"));
    m.def("compute_u8", &compute_u8, nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"));
}
