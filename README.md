# tensorstats

Fast exact central moment computation for NumPy arrays. Computes mean, variance, 3rd and 4th central moments in a single pass — substantially faster than NumPy for repeated computation on fixed shapes.

## Installation

```bash
pip install -e .
```

Requires a C++17 compiler. Builds with AVX2 on GCC/Clang (`-O3 -march=native`) and MSVC (`/O2 /arch:AVX2`).

## Usage

```python
import tensorstats as ts
import numpy as np

sc = ts.StatsComputer(
    shape=(64, 64, 3),
    axes=[None, (0, 1)],   # global + per-channel reductions
    stride=(4, 4, 1),      # subsample H and W by 4
    grid=(4, 4, 2),        # 16x16 spatial grid, 4 channel cells
)

result = sc.compute(hsv_frame)
result["global"]   # (4,)          [mean, variance, m3, m4] over all pixels
result["0,1"]      # (3, 4)        per-channel moments (H/S/V or B/G/R)
result["grid"]     # (16,16,4,4)   spatial grid moments

# Derive higher-level statistics from raw moments as needed:
var      = result["global"][1]
std      = np.sqrt(var)
skewness = result["global"][2] / std**3
kurtosis = result["global"][3] / var**2
```

Call `sc.compute()` on every frame — it reuses internal buffers.

## API

### `ts.StatsComputer`

```python
sc = ts.StatsComputer(
    shape,          # tuple — input shape, fixed for all compute() calls
    axes=None,      # None=global, int, tuple, or list e.g. [None, (0,1)]
    stride=None,    # None/1=no stride, int, or per-axis tuple e.g. (4,4,1)
    grid=None,      # None=no grid, int k (2^k cells/axis), or tuple e.g. (4,4,2)
    n_moments=4,    # 1–4
)
result = sc.compute(arr)
```

**Result keys** depend on the `axes` and `grid` arguments:

| Key | Shape | When |
|-----|-------|------|
| `"global"` | `(n_moments,)` | `axes` includes `None` |
| `"<a>,<b>"` | `(*kept_shape, n_moments)` | `axes` includes a tuple |
| `"grid"` | `(*cell_shape, n_moments)` | `grid` is set |

**Moments layout** (last axis, same for all outputs):

```
[..., 0]  mean
[..., 1]  variance
[..., 2]  3rd central moment
[..., 3]  4th central moment
```

**Grid parameters:** `grid[d] = k` means `2^k` cells along axis `d`. `grid=(4,4,2)` on a `(H,W,C)` array gives a 16×16 spatial grid with 4 channel cells. For `C=3`, cell boundaries are `c * 4 / 3` (integer division): channels 0→cell0, 1→cell1, 2→cell2, cell3 empty.

**Supported dtypes:** uint8, float32, float64. Other dtypes are cast to float64.

## Running the tests

```bash
# From the repo root
python -m pytest tests/ -v -s
```

All tests use `ts.StatsComputer` — the sole public interface.

## Performance

Benchmark on Linux (GCC -O3 -march=native), `axes=[None,(0,1)], grid=(4,4,2)`, `StatsComputer.compute()` p50 latency:

| Shape | dtype | stride | latency |
|-------|-------|--------|---------|
| 64×64×3 | uint8 | 1 | 0.115ms |
| 64×64×3 | uint8 | 4 | 0.118ms |
| 64×64×3 | float64 | 1 | 0.120ms |
| 64×64×3 | float64 | 4 | 0.059ms |
| 128×128×3 | uint8 | 1 | 0.414ms |
| 128×128×3 | uint8 | 4 | 0.460ms |
| 128×128×3 | float64 | 4 | 0.202ms |

`StatsComputer` is ~2× faster than constructing fresh each call, because the grid path retains its cell-index array and output buffer across calls.

## Implementation notes

**uint8 histogram path** (global and per-channel reductions): builds a `hist[256]` then computes moments with 256 FMA operations instead of N float operations. Uses 4-way parallel counters. Faster than the float path for N ≥ 256 pixels.

**Grid path**: pixel-outer scatter into per-cell accumulators using a precomputed `cell_of[]` (int16) flat index array. Eliminates per-pixel index arithmetic. Two passes: sums then moments. The `cell_of[]` array is computed once at construction and reused every frame.

**Stride**: subsamples the input before computing moments. The library computes exact moments over the subsampled pixels — it does not interpolate or approximate beyond the sampling itself.
