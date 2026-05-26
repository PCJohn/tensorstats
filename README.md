# tensorstats

Fast central moment computation for NumPy arrays. Computes mean, variance, skewness, and kurtosis in a single library call — substantially faster than NumPy for repeated computation on fixed shapes.

## Installation

```bash
pip install -e .
```

Requires a C++17 compiler. Builds with AVX2 on GCC/Clang (`-O3 -march=native`) and MSVC (`/O2 /arch:AVX2`).

## Quick start

```python
import tensorstats as ts
import numpy as np

# Stateless convenience function
arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
result = ts.compute(arr, axes=[None, (0, 1)], stride=(4, 4, 1), grid=(4, 4, 2))

result["global"]   # (4,)        — global moments over all pixels
result["0,1"]      # (3, 4)      — per-channel moments (one row per channel)
result["grid"]     # (16,16,4,4) — 16×16 spatial grid, 4 channel cells, 4 moments

# Stateful — retain buffers across calls (~2x faster on the grid path)
sc = ts.StatsComputer(
    shape=(64, 64, 3),
    axes=[None, (0, 1)],
    stride=(4, 4, 1),
    grid=(4, 4, 2),
)
for frame in frames:
    result = sc.compute(frame)
    # result["grid"] is a VIEW into retained memory — copy if keeping past next call
    grid_copy = result["grid"].copy()
```

## Output convention

All results use **moments-last** layout: shape is `(*reduction_shape, n_moments)`.

| Key | Shape | When |
|-----|-------|------|
| `"global"` | `(n_moments,)` | `axes` contains `None` |
| `"0,1"` | `(C, n_moments)` | `axes` contains `(0,1)` |
| `"grid"` | `(*cell_shape, n_moments)` | `grid` is set |

Moment layout (last axis):

```
[..., 0]  mean
[..., 1]  variance (2nd central moment)
[..., 2]  3rd central moment
[..., 3]  4th central moment
```

Derived quantities:

```python
std      = np.sqrt(result["global"][1])
skewness = result["global"][2] / std**3
kurtosis = result["global"][3] / result["global"][1]**2
```

## API

### `ts.StatsComputer` — stateful, preferred for repeated calls

```python
sc = ts.StatsComputer(
    shape,       # tuple — fixed input shape, e.g. (64, 64, 3)
    axes=None,   # None=global, int, tuple, or list of specs e.g. [None, (0,1)]
    stride=None, # None/1=no stride, int, or per-axis tuple e.g. (4,4,1)
    grid=None,   # None=no grid, int k (2^k cells/axis), or tuple e.g. (4,4,2)
    n_moments=4, # 1–4
)
result = sc.compute(arr)      # arr must match shape; returns dict[str, ndarray]
sc.set_shape((128, 128, 3))   # change shape, reallocates grid buffers
sc.set_grid((3, 3, 2))        # change grid config only
```

### `ts.compute` — stateless convenience wrapper

```python
result = ts.compute(arr, axes=None, stride=None, n_moments=4, grid=None)
```

Creates a temporary `StatsComputer` and returns independent copies of all outputs. Use `StatsComputer` directly when calling in a loop.

### Grid parameters

`grid[d] = k` means `2**k` cells along axis `d`. `grid=(4,4,2)` on a `(H,W,C)` array gives `16×16` spatial cells and `4` channel cells (only 3 populated for `C=3`).

Cell boundaries use integer division: `cell_of[coord, d] = coord * n_cells[d] / shape[d]`. For `C=3` with `n_cells[C]=4`: channel 0→cell 0, channel 1→cell 1, channel 2→cell 2, cell 3 empty.

## Performance

Benchmark on Linux (GCC -O3 -march=native), `axes=[None,(0,1)], grid=(4,4,2)`, p50 latency:

| Shape | dtype | stride | `ts.compute` | `StatsComputer` | speedup |
|-------|-------|--------|-------------|-----------------|---------|
| 64×64×3 | uint8 | 1 | 0.216ms | 0.129ms | 1.67× |
| 64×64×3 | uint8 | 2 | 0.328ms | 0.252ms | 1.30× |
| 64×64×3 | uint8 | 4 | 0.237ms | 0.143ms | 1.66× |
| 64×64×3 | float64 | 1 | 0.218ms | 0.135ms | 1.62× |
| 64×64×3 | float64 | 2 | 0.270ms | 0.171ms | 1.58× |
| 64×64×3 | float64 | 4 | 0.170ms | 0.084ms | 2.03× |
| 128×128×3 | uint8 | 1 | 0.765ms | 0.471ms | 1.63× |
| 128×128×3 | uint8 | 2 | 1.258ms | 0.974ms | 1.29× |
| 128×128×3 | uint8 | 4 | 0.833ms | 0.541ms | 1.54× |
| 128×128×3 | float64 | 1 | 0.813ms | 0.509ms | 1.60× |
| 128×128×3 | float64 | 2 | 0.965ms | 0.663ms | 1.46× |
| 128×128×3 | float64 | 4 | 0.588ms | 0.293ms | 2.01× |

The `StatsComputer` speedup comes from two sources:
- **Retained `cell_of[]` array** (int16, precomputed flat cell indices) — eliminates per-pixel index arithmetic in the scatter loops
- **Retained output buffer** returned as a zero-copy view — eliminates `new double[]` allocation and nanobind ndarray construction per call

## Implementation notes

**uint8 histogram path** (global and per-channel reductions): builds a `hist[256]` then computes moments with 256 FMAs instead of N float operations. Uses 4-way parallel counters to reduce write-after-write stalls. Faster than two-pass float for `N >= 256` pixels.

**Grid path**: single forward pass over all pixels scattering into per-cell accumulators via `cell_of[i]` lookup, followed by a finalisation pass over cells. The precomputed `cell_of[]` (int16, 24KB for 64×64×3) fits in L1 cache and is retained across calls by `StatsComputer`.

**Stride**: subsamples the input before computation. `stride=(4,4,1)` on `(64,64,3)` reads every 4th row and column but all channels, giving a 16× reduction in pixel count with proportional speed increase.
