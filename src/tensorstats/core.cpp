#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <string>
#include <cstdint>

// Portable no-alias hint
#if defined(_MSC_VER)
#  define TS_RESTRICT __restrict
#else
#  define TS_RESTRICT __restrict__
#endif

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// tensorstats — fast two-pass central moment computation
//
// Output convention: moments-LAST. All output arrays have shape
//   (*reduction_output_shape, n_moments)
// e.g. global → (n_moments,), per-channel → (C, n_moments),
//      grid   → (g0_cells, g1_cells, ..., n_moments)
//
// Algorithm: two-pass, numerically stable:
//   Pass 1: mean per bucket  — pure sum, SIMD-vectorized
//   Pass 2: d=x-mu, accumulate d^2, d^2*d, d^2*d^2 — 4 FMAs/element
//
// Specialised paths (selected at runtime):
//   Global    — single accumulator, straight loop, AVX2-vectorized
//   LastAxis  — stride-C column loops (no modulo), AVX2-vectorized
//   General   — iterative (flat_idx, bucket) pair enumeration, arbitrary axes
//   Grid      — power-of-2 cell grid; iterative per-cell enumeration;
//               histogram path when pixels/cell >= 256 (uint8 only),
//               direct two-pass otherwise. Crossover benchmarked at ~256px/cell.
//
// uint8 histogram path (Global + LastAxis + large Grid cells):
//   Replaces N float FMAs with N integer increments + 256 float FMAs.
//   4-way parallel counters (h0/h1/h2/h3) reduce write-after-write stalls.
//   Mean from exact integer sum — no float rounding in pass 1.
//
// No std::function used anywhere — all iteration is explicit to avoid
// virtual-dispatch overhead (~5x on grid collect, ~1.6x on pair enumeration).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// uint8 histogram helpers
// ---------------------------------------------------------------------------

static void build_hist4(const uint8_t* TS_RESTRICT d, int64_t n,
                         int64_t* TS_RESTRICT hist)
{
    int64_t h0[256]={}, h1[256]={}, h2[256]={}, h3[256]={};
    int64_t i=0, n4=(n>>2)<<2;
    for(;i<n4;i+=4){ h0[d[i]]++;h1[d[i+1]]++;h2[d[i+2]]++;h3[d[i+3]]++; }
    for(;i<n;++i) h0[d[i]]++;
    for(int v=0;v<256;++v) hist[v]=h0[v]+h1[v]+h2[v]+h3[v];
}

static void moments_from_hist(const int64_t* TS_RESTRICT hist, int64_t n,
                               double &mu, double &m2, double &m3, double &m4)
{
    int64_t s1=0;
    for(int v=0;v<256;++v) s1+=hist[v]*(int64_t)v;
    mu=(double)s1/(double)n;
    m2=m3=m4=0.0;
    for(int v=0;v<256;++v){
        if(!hist[v]) continue;
        double x=(double)v-mu, x2=x*x, h=(double)hist[v];
        m2+=h*x2; m3+=h*x2*x; m4+=h*x2*x2;
    }
    m2/=n; m3/=n; m4/=n;
}

static void global_u8_hist(const uint8_t* TS_RESTRICT d, int64_t n, int64_t step,
                            double &mu, double &m2, double &m3, double &m4)
{
    int64_t hist[256]={};
    if(step==1){
        build_hist4(d,n,hist);
        moments_from_hist(hist,n,mu,m2,m3,m4);
    } else {
        // 4-way parallel strided histogram: same write-conflict reduction as
        // build_hist4 but with step spacing between lanes (~1.1x faster).
        int64_t h0[256]={},h1[256]={},h2[256]={},h3[256]={};
        int64_t step4=step*4, ns=0;
        int64_t i=0, n4=(n/step4)*step4;
        for(;i<n4;i+=step4){
            h0[d[i]]++;h1[d[i+step]]++;h2[d[i+step*2]]++;h3[d[i+step*3]]++;
            ns+=4;
        }
        for(;i<n;i+=step){ h0[d[i]]++;ns++; }
        for(int v=0;v<256;++v) hist[v]=h0[v]+h1[v]+h2[v]+h3[v];
        moments_from_hist(hist,ns,mu,m2,m3,m4);
    }
}

static void last_axis_u8_hist(const uint8_t* TS_RESTRICT d,
                               int64_t HW, int64_t C, int64_t sr, int64_t sc,
                               std::vector<double> &mu, std::vector<double> &m2,
                               std::vector<double> &m3, std::vector<double> &m4)
{
    for(int64_t c=0;c<C;c+=sc){
        int64_t hist[256]={}, ns=0;
        for(int64_t r=0;r<HW;r+=sr){ hist[d[r*C+c]]++;ns++; }
        moments_from_hist(hist,ns,mu[c],m2[c],m3[c],m4[c]);
    }
}

// ---------------------------------------------------------------------------
// Sampled flat index list — only used for non-uniform strides (rare path)
// ---------------------------------------------------------------------------
static std::vector<int64_t> sampled_indices(const std::vector<int64_t> &shape,
                                             const std::vector<int64_t> &stride)
{
    int ndim=(int)shape.size();
    int64_t cap=1;
    for(int d=0;d<ndim;++d) cap*=(shape[d]+stride[d]-1)/stride[d];
    std::vector<int64_t> result; result.reserve(cap);

    // C-order flat strides
    std::vector<int64_t> fs(ndim,1);
    for(int d=ndim-2;d>=0;--d) fs[d]=fs[d+1]*shape[d+1];

    // Explicit coordinate iteration — no std::function
    std::vector<int64_t> coords(ndim,0);
    int64_t flat=0;
    while(true){
        result.push_back(flat);
        int d=ndim-1;
        while(d>=0){
            coords[d]+=stride[d]; flat+=stride[d]*fs[d];
            if(coords[d]<shape[d]) break;
            flat-=coords[d]*fs[d]; coords[d]=0; --d;
        }
        if(d<0) break;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Inner loop kernels
// ---------------------------------------------------------------------------
template<typename T>
static void global_pass(const T* TS_RESTRICT data, int64_t n, int64_t step,
                         double &mu, double &m2, double &m3, double &m4)
{
    int64_t ns=(n+step-1)/step;
    double s=0;
    for(int64_t i=0;i<n;i+=step) s+=(double)data[i];
    mu=s/(double)ns; m2=m3=m4=0;
    for(int64_t i=0;i<n;i+=step){ double d=(double)data[i]-mu,d2=d*d; m2+=d2;m3+=d2*d;m4+=d2*d2; }
    double inv=1.0/(double)ns; m2*=inv; m3*=inv; m4*=inv;
}

template<typename T>
static void global_pass_idx(const T* TS_RESTRICT data,
                              const std::vector<int64_t> &idx,
                              double &mu, double &m2, double &m3, double &m4)
{
    int64_t ns=(int64_t)idx.size(); double s=0;
    for(int64_t i:idx) s+=(double)data[i];
    mu=s/(double)ns; m2=m3=m4=0;
    for(int64_t i:idx){ double d=(double)data[i]-mu,d2=d*d; m2+=d2;m3+=d2*d;m4+=d2*d2; }
    double inv=1.0/(double)ns; m2*=inv; m3*=inv; m4*=inv;
}

template<typename T>
static void last_axis_pass(const T* TS_RESTRICT data,
                            int64_t HW, int64_t C, int64_t sr, int64_t sc,
                            std::vector<double> &mu, std::vector<double> &m2,
                            std::vector<double> &m3, std::vector<double> &m4)
{
    int64_t nrows=(HW+sr-1)/sr; double inv=1.0/(double)nrows;
    for(int64_t c=0;c<C;c+=sc){
        double s=0;
        for(int64_t r=0;r<HW;r+=sr) s+=(double)data[r*C+c];
        mu[c]=s*inv; double s2=0,s3=0,s4=0;
        for(int64_t r=0;r<HW;r+=sr){ double d=(double)data[r*C+c]-mu[c],d2=d*d; s2+=d2;s3+=d2*d;s4+=d2*d2; }
        m2[c]=s2*inv; m3[c]=s3*inv; m4[c]=s4*inv;
    }
}

template<typename T>
static void general_pass(const T* TS_RESTRICT data,
                          const std::vector<std::pair<int64_t,int64_t>> &pairs,
                          int64_t n_buckets,
                          std::vector<double> &mu, std::vector<double> &m2,
                          std::vector<double> &m3, std::vector<double> &m4)
{
    std::fill(mu.begin(),mu.end(),0.0);
    std::vector<int64_t> counts(n_buckets,0);
    for(auto [fi,b]:pairs){ mu[b]+=(double)data[fi]; counts[b]++; }
    for(int64_t b=0;b<n_buckets;++b)
        mu[b]=counts[b]>0?mu[b]/(double)counts[b]:0.0;
    std::fill(m2.begin(),m2.end(),0.0);
    std::fill(m3.begin(),m3.end(),0.0);
    std::fill(m4.begin(),m4.end(),0.0);
    for(auto [fi,b]:pairs){ double d=(double)data[fi]-mu[b],d2=d*d; m2[b]+=d2;m3[b]+=d2*d;m4[b]+=d2*d2; }
    for(int64_t b=0;b<n_buckets;++b) if(counts[b]>0){
        double inv=1.0/(double)counts[b]; m2[b]*=inv; m3[b]*=inv; m4[b]*=inv;
    }
}

// ---------------------------------------------------------------------------
// Axis spec
// ---------------------------------------------------------------------------
struct AxisSpec {
    std::vector<int>     reduce_axes;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_strides;
    int64_t              acc_size    = 1;
    bool                 is_last_dim = false;
    int64_t              last_dim_C  = 1;
};

static AxisSpec make_axis_spec(const std::vector<int> &axes,
                               const std::vector<int64_t> &shape)
{
    int ndim=(int)shape.size();
    AxisSpec s;
    for(int a:axes){ int na=(a<0)?a+ndim:a; if(na<0||na>=ndim) throw std::out_of_range("axis out of range"); s.reduce_axes.push_back(na); }
    std::sort(s.reduce_axes.begin(),s.reduce_axes.end());
    s.reduce_axes.erase(std::unique(s.reduce_axes.begin(),s.reduce_axes.end()),s.reduce_axes.end());
    s.acc_size=1;
    for(int d=0;d<ndim;++d){
        bool red=std::find(s.reduce_axes.begin(),s.reduce_axes.end(),d)!=s.reduce_axes.end();
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

// Build (flat_idx, bucket_idx) pairs for all sampled elements.
// Fully iterative — no std::function, no recursion overhead (~1.6x faster).
static std::vector<std::pair<int64_t,int64_t>> make_pairs(
    const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
    const std::vector<int> &reduce_axes, const std::vector<int64_t> &out_strides)
{
    int ndim=(int)shape.size();

    // Precompute per-dim: is_reduced, output_dim_index, C-order flat stride
    std::vector<bool>    is_red(ndim, false);
    std::vector<int64_t> odim(ndim, -1);   // output dim index for non-reduced dims
    std::vector<int64_t> fs(ndim, 1);      // C-order flat strides

    for(int a:reduce_axes) is_red[a]=true;
    int od=0; for(int d=0;d<ndim;++d) if(!is_red[d]) odim[d]=od++;
    for(int d=ndim-2;d>=0;--d) fs[d]=fs[d+1]*shape[d+1];

    int64_t cap=1;
    for(int d=0;d<ndim;++d) cap*=(shape[d]+stride[d]-1)/stride[d];
    std::vector<std::pair<int64_t,int64_t>> pairs;
    pairs.reserve(cap);

    // Explicit coordinate iteration
    std::vector<int64_t> coords(ndim,0);
    int64_t flat=0, bkt=0;
    while(true){
        pairs.push_back({flat,bkt});
        int d=ndim-1;
        while(d>=0){
            coords[d]+=stride[d];
            flat+=stride[d]*fs[d];
            if(!is_red[d] && odim[d]>=0) bkt+=stride[d]*out_strides[odim[d]];
            if(coords[d]<shape[d]) break;
            flat-=coords[d]*fs[d];
            if(!is_red[d] && odim[d]>=0) bkt-=coords[d]*out_strides[odim[d]];
            coords[d]=0; --d;
        }
        if(d<0) break;
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// Pack — moments-LAST: (*out_shape, n_moments)
// Flat layout: ptr[cell_flat * n_moments + k]
// ---------------------------------------------------------------------------
static nb::ndarray<nb::numpy,double> pack(
    const std::vector<double> &mu, const std::vector<double> &m2,
    const std::vector<double> &m3, const std::vector<double> &m4,
    const std::vector<int64_t> &out_shape, int64_t nacc, int n_moments)
{
    std::vector<size_t> sh;
    for(auto d:out_shape) sh.push_back((size_t)d);
    sh.push_back((size_t)n_moments);
    auto *ptr=new double[(size_t)nacc*(size_t)n_moments];
    for(int64_t i=0;i<nacc;++i){
        if(n_moments>=1) ptr[i*n_moments+0]=mu[i];
        if(n_moments>=2) ptr[i*n_moments+1]=m2[i];
        if(n_moments>=3) ptr[i*n_moments+2]=m3[i];
        if(n_moments>=4) ptr[i*n_moments+3]=m4[i];
    }
    nb::capsule own(ptr,[](void*p)noexcept{delete[]static_cast<double*>(p);});
    return nb::ndarray<nb::numpy,double>(ptr,sh.size(),sh.data(),own);
}

// ---------------------------------------------------------------------------
// Grid computation
//
// grid[d] = log2 of number of cells along axis d (0 = no subdivision).
//   0 → 1 cell,  1 → 2 cells,  k → 2^k cells
//
// Cell boundaries: lo = cell_idx * shape[d] / n_cells[d]  (works for any shape)
//
// uint8 histogram path: cells with >= HIST_THRESHOLD pixels (benchmarked crossover).
// Direct two-pass: small cells and all float types.
//
// Flat index collection is fully iterative (no std::function) — ~5.8x faster
// than the recursive lambda approach, measured on 8x8 grid of 64x64 image.
// ---------------------------------------------------------------------------

static constexpr int64_t HIST_THRESHOLD = 256;

template<typename T>
static void compute_cell_direct(const T* TS_RESTRICT data,
                                  const std::vector<int64_t> &idx,
                                  double &mu, double &m2, double &m3, double &m4)
{
    int64_t n=(int64_t)idx.size(); double s=0;
    for(int64_t fi:idx) s+=(double)data[fi];
    mu=s/(double)n; m2=m3=m4=0;
    for(int64_t fi:idx){ double d=(double)data[fi]-mu,d2=d*d; m2+=d2;m3+=d2*d;m4+=d2*d2; }
    double inv=1.0/(double)n; m2*=inv; m3*=inv; m4*=inv;
}

template<typename T>
static nb::ndarray<nb::numpy,double> compute_grid_typed(
    const T* TS_RESTRICT data,
    const std::vector<int64_t> &shape,
    const std::vector<int>     &grid,
    int n_moments)
{
    int ndim=(int)shape.size();

    std::vector<int64_t> n_cells(ndim);
    for(int d=0;d<ndim;++d) n_cells[d]=(int64_t)1<<grid[d];

    int64_t total_cells=1;
    for(int d=0;d<ndim;++d) total_cells*=n_cells[d];

    // C-order flat strides of input array
    std::vector<int64_t> fs(ndim,1);
    for(int d=ndim-2;d>=0;--d) fs[d]=fs[d+1]*shape[d+1];

    // Allocate output: (*n_cells, n_moments)
    std::vector<size_t> out_sh;
    for(int d=0;d<ndim;++d) out_sh.push_back((size_t)n_cells[d]);
    out_sh.push_back((size_t)n_moments);
    auto *ptr=new double[(size_t)total_cells*(size_t)n_moments];

    // Preallocate flat_indices buffer — reused across cells
    // Max cell size = total elements / total_cells (when all cells equal size)
    int64_t total=1; for(auto s:shape) total*=s;
    int64_t max_cell_size = (total + total_cells - 1) / total_cells * 2;
    std::vector<int64_t> flat_indices;
    flat_indices.reserve((size_t)max_cell_size);

    // Precompute cell boundaries per axis per cell index
    // lo[d][ci] = ci * shape[d] / n_cells[d]
    std::vector<std::vector<int64_t>> lo(ndim), hi(ndim);
    for(int d=0;d<ndim;++d){
        lo[d].resize(n_cells[d]); hi[d].resize(n_cells[d]);
        for(int64_t ci=0;ci<n_cells[d];++ci){
            lo[d][ci] = ci * shape[d] / n_cells[d];
            hi[d][ci] = (ci+1) * shape[d] / n_cells[d];
        }
    }

    // Iterate over all cells
    std::vector<int64_t> cell_coords(ndim,0);

    for(int64_t cell=0; cell<total_cells; ++cell){
        // Collect flat indices for this cell — fully iterative
        flat_indices.clear();
        std::vector<int64_t> elem_coords(ndim);
        for(int d=0;d<ndim;++d) elem_coords[d]=lo[d][cell_coords[d]];

        int64_t elem_flat=0;
        for(int d=0;d<ndim;++d) elem_flat+=elem_coords[d]*fs[d];

        // Iterate elements within the cell bounds
        while(true){
            flat_indices.push_back(elem_flat);
            int d=ndim-1;
            while(d>=0){
                ++elem_coords[d]; elem_flat+=fs[d];
                if(elem_coords[d]<hi[d][cell_coords[d]]) break;
                elem_flat-=(elem_coords[d]-lo[d][cell_coords[d]])*fs[d];
                elem_coords[d]=lo[d][cell_coords[d]]; --d;
            }
            if(d<0) break;
        }

        double mu=0,m2=0,m3=0,m4=0;
        if constexpr (std::is_same_v<T, uint8_t>) {
            int64_t n=(int64_t)flat_indices.size();
            if(n>=HIST_THRESHOLD){
                int64_t hist[256]={};
                for(int64_t fi:flat_indices) hist[data[fi]]++;
                moments_from_hist(hist,n,mu,m2,m3,m4);
            } else {
                compute_cell_direct<T>(data,flat_indices,mu,m2,m3,m4);
            }
        } else {
            compute_cell_direct<T>(data,flat_indices,mu,m2,m3,m4);
        }

        double *out=ptr+cell*n_moments;
        if(n_moments>=1) out[0]=mu;
        if(n_moments>=2) out[1]=m2;
        if(n_moments>=3) out[2]=m3;
        if(n_moments>=4) out[3]=m4;

        // Advance cell_coords
        for(int d=ndim-1;d>=0;--d){
            if(++cell_coords[d]<n_cells[d]) break;
            cell_coords[d]=0;
        }
    }

    nb::capsule own(ptr,[](void*p)noexcept{delete[]static_cast<double*>(p);});
    return nb::ndarray<nb::numpy,double>(ptr,out_sh.size(),out_sh.data(),own);
}

// ---------------------------------------------------------------------------
// Main typed entry point
// ---------------------------------------------------------------------------
template<typename T>
static nb::dict compute_typed(
    const T *data, const std::vector<int64_t> &shape,
    const std::vector<std::vector<int>> &axes_list,
    const std::vector<int64_t> &stride, int n_moments)
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
                if(!has_stride||uniform_stride)
                    global_u8_hist(data,total,uniform_stride?s0:1,mu,m2,m3,m4);
                else { auto idx=sampled_indices(shape,stride); global_pass_idx<T>(data,idx,mu,m2,m3,m4); }
            } else {
                if(!has_stride) global_pass<T>(data,total,1,mu,m2,m3,m4);
                else if(uniform_stride) global_pass<T>(data,total,s0,mu,m2,m3,m4);
                else { auto idx=sampled_indices(shape,stride); global_pass_idx<T>(data,idx,mu,m2,m3,m4); }
            }
            std::vector<size_t> sh={(size_t)n_moments};
            auto *p=new double[n_moments];
            if(n_moments>=1)p[0]=mu; if(n_moments>=2)p[1]=m2;
            if(n_moments>=3)p[2]=m3; if(n_moments>=4)p[3]=m4;
            nb::capsule own(p,[](void*x)noexcept{delete[]static_cast<double*>(x);});
            result["global"]=nb::ndarray<nb::numpy,double>(p,1,sh.data(),own);

        } else {
            AxisSpec spec=make_axis_spec(raw_axes,shape);
            int64_t nacc=spec.acc_size;
            std::vector<double> mu(nacc),m2(nacc),m3(nacc),m4(nacc);

            int ndim_red=(int)spec.reduce_axes.size();
            bool simple_hw_stride=true;
            if(ndim_red>1&&has_stride){
                int64_t sr_check=stride[spec.reduce_axes[0]];
                for(int i=1;i<ndim_red;++i)
                    if(stride[spec.reduce_axes[i]]!=sr_check){simple_hw_stride=false;break;}
                if(simple_hw_stride&&ndim_red>1)
                    for(int i=1;i<ndim_red;++i)
                        if(stride[spec.reduce_axes[i]]!=1){simple_hw_stride=false;break;}
            }

            if(spec.is_last_dim){
                int64_t HW=total/spec.last_dim_C;
                if constexpr (std::is_same_v<T, uint8_t>) {
                    if(!has_stride)
                        last_axis_u8_hist(data,HW,spec.last_dim_C,1,1,mu,m2,m3,m4);
                    else if(simple_hw_stride&&ndim_red==1)
                        last_axis_u8_hist(data,HW,spec.last_dim_C,
                            stride[spec.reduce_axes[0]],stride[ndim-1],mu,m2,m3,m4);
                    else { auto p=make_pairs(shape,stride,spec.reduce_axes,spec.out_strides); general_pass<T>(data,p,nacc,mu,m2,m3,m4); }
                } else if(!has_stride)
                    last_axis_pass<T>(data,HW,spec.last_dim_C,1,1,mu,m2,m3,m4);
                else if(simple_hw_stride&&ndim_red==1)
                    last_axis_pass<T>(data,HW,spec.last_dim_C,stride[spec.reduce_axes[0]],stride[ndim-1],mu,m2,m3,m4);
                else { auto p=make_pairs(shape,stride,spec.reduce_axes,spec.out_strides); general_pass<T>(data,p,nacc,mu,m2,m3,m4); }
            } else {
                auto p=make_pairs(shape,stride,spec.reduce_axes,spec.out_strides);
                general_pass<T>(data,p,nacc,mu,m2,m3,m4);
            }

            std::string key;
            for(int i=0;i<(int)raw_axes.size();++i){ if(i)key+=","; key+=std::to_string(raw_axes[i]); }
            result[key.c_str()]=pack(mu,m2,m3,m4,spec.out_shape,nacc,n_moments);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public bindings
// ---------------------------------------------------------------------------
using ArrF64=nb::ndarray<nb::numpy,double, nb::c_contig,nb::device::cpu>;
using ArrF32=nb::ndarray<nb::numpy,float,  nb::c_contig,nb::device::cpu>;
using ArrU8 =nb::ndarray<nb::numpy,uint8_t,nb::c_contig,nb::device::cpu>;

#define MAKE_ENTRY(name, T, ArrT)                                              \
nb::dict name(ArrT arr,                                                        \
              std::vector<std::vector<int>> axes,                              \
              std::vector<int64_t> stride, int n_moments,                      \
              std::vector<int> grid){                                           \
    int ndim=(int)arr.ndim();                                                  \
    std::vector<int64_t> shape(ndim);                                          \
    for(int d=0;d<ndim;++d) shape[d]=arr.shape(d);                            \
    if((int)stride.size()!=ndim)                                               \
        throw std::invalid_argument("stride length must match ndim");          \
    nb::dict result=compute_typed<T>(arr.data(),shape,axes,stride,n_moments);  \
    if(!grid.empty()){                                                         \
        if((int)grid.size()!=ndim)                                             \
            throw std::invalid_argument("grid length must match ndim");        \
        result["grid"]=compute_grid_typed<T>(arr.data(),shape,grid,n_moments); \
    }                                                                          \
    return result;                                                             \
}

MAKE_ENTRY(compute_f64, double,  ArrF64)
MAKE_ENTRY(compute_f32, float,   ArrF32)
MAKE_ENTRY(compute_u8,  uint8_t, ArrU8)

NB_MODULE(tensorstats_core, m){
    m.doc()=R"(
Fast two-pass central moment computation. Moments-LAST output convention.

Output shapes (axes reduction):
  "global"  → (n_moments,)
  "0,1"     → (C, n_moments)       for axes=(0,1) on (H,W,C)
  "0"       → (W, C, n_moments)    for axis=0 on (H,W,C)

Output shape (grid):
  "grid"    → (*cell_shape, n_moments)
              e.g. (4,4,1,n_moments) for grid=(2,2,0) on (H,W,C)

grid[d] = log2(cells along axis d). 0=no subdivision, 3=8 cells.
uint8: histogram path for global, last-axis, and grid cells >= 256 pixels.
All iteration is explicit (no std::function) for minimal call overhead.
    )";
    m.def("compute_f64",&compute_f64,nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"),nb::arg("grid"));
    m.def("compute_f32",&compute_f32,nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"),nb::arg("grid"));
    m.def("compute_u8", &compute_u8, nb::arg("arr"),nb::arg("axes"),nb::arg("stride"),nb::arg("n_moments"),nb::arg("grid"));
}
