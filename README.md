# tensorstats

Fast exact central moment computation for NumPy arrays. Computes mean, variance,
3rd and 4th central moments in a single pass — substantially faster than NumPy
for repeated computation on fixed shapes.

- **Exact**: same numerical result as NumPy (to floating-point precision)
- **Stateful**: `StatsComputer` retains internal buffers across calls (~2× faster than fresh construction per call)
- **Flexible**: arbitrary axis reductions, spatial grid cells, per-axis stride
- **Fast uint8 path**: histogram-based moments — 256 FMAs instead of N float ops
- **Cross-platform**: GCC, Clang, MSVC (Windows); tested on Linux, Mac, Windows


## Installation

```bash
pip install -e .
```

Requires a C++17 compiler. Builds with `-O3 -march=native -ffast-math` on GCC/Clang
and `/O2 /arch:AVX2 /fp:fast` on MSVC.


## Quick start

```python
import tensorstats as ts
import numpy as np

# Construct once for a fixed shape and config
sc = ts.StatsComputer(
    shape=(64, 64, 3),
    axes=[None, (0, 1)],  # global + per-channel reductions
    stride=(4, 4, 1),     # subsample H and W by 4, keep all channels (grid too)
    grid=[(4, 4, 2), (5, 5, 0)],  # a grid pyramid: two resolutions, one pass
)

# Call on every frame — reuses internal buffers
result = sc.compute(hsv_frame)

result["global"]  # (4,)           [mean, variance, m3, m4] over all pixels
result["0,1"]     # (3, 4)         per-channel moments (one row per channel)
result["grid_0"]  # (16,16,4,4)    level-0 grid moments
result["grid_1"]  # (32,32,1,4)    level-1 grid moments

# Derive higher-level quantities from the raw central moments
var      = result["global"][1]
std      = np.sqrt(var)
skewness = result["global"][2] / std**3
kurtosis = result["global"][3] / var**2
```


## API

### `ts.StatsComputer`

The sole public interface.

```python
sc = ts.StatsComputer(
    shape,         # tuple — every compute() call must use this exact shape
    axes=None,     # None=global, int, tuple, or list e.g. [None, (0,1)]
    stride=None,   # None/1=no stride, int, or per-axis tuple e.g. (4,4,1)
    grid=None,     # None=no grid, int k (2^k cells/axis), a per-axis tuple
                   #   e.g. (4,4,2), or a LIST of tuples for a grid pyramid
    n_moments=4,   # 1–4
)
result = sc.compute(arr)  # dict[str, np.ndarray]
```

**Result keys:**

| Key | Shape | Produced when |
|-----|-------|---------------|
| `"global"` | `(n_moments,)` | `axes` includes `None` |
| `"0,1"` | `(C, n_moments)` | `axes` includes `(0,1)` |
| `"grid_{i}"` | `(*cell_shape, n_moments)` | one per grid level, `i = 0 … K-1` |

Key name for an axis tuple is the comma-joined axis indices, e.g. `"0,1"` for `axes=(0,1)`.

**Moments layout** — last axis, same for every output:

```
[..., 0]  mean
[..., 1]  variance  (= 2nd central moment)
[..., 2]  3rd central moment
[..., 3]  4th central moment
```

Callers derive std, skewness, kurtosis from these raw moments. The library
returns exact central moments and does not make choices about which derived
quantities are wanted.

**Grid:** `grid[d] = k` → `2^k` cells along axis `d`. Example: `grid=(4,4,2)` on
`(H,W,C)` gives a 16×16 spatial grid with 4 channel cells. Cell boundaries use
integer floor-division: `cell[coord, d] = coord * n_cells[d] / shape[d]`.
For `C=3` and 4 channel cells: H→cell0, S→cell1, V→cell2, cell3 empty.

Pass a **list** of specs for a *grid pyramid* — multiple resolutions computed in a
single pass over the tensor, e.g. `grid=[(2,2,0),(3,3,0),(4,4,0)]`. Each level is
returned under its own key `grid_0 … grid_{K-1}` in list order. A single tuple is
the `K=1` case and yields `grid_0`. Adding levels adds accumulators, not passes:
the tensor is read once (all-histogram path) or twice (any direct level),
independent of the number of levels.

**Stride:** subsamples the input. `stride=(4,4,1)` on `(H,W,C)` visits every 4th row
and column, all channels. The library computes exact moments over the subsampled
pixels — no interpolation. Stride applies to the grid path too: each cell's
moments are the exact moments over the visited pixels of that cell, with cell
boundaries kept at full resolution (so it is *subsample-then-skip*, not
subsample-then-rebin). For a grid, stride is the main latency lever.

**Supported dtypes:** uint8, float32, float64. Other dtypes are cast to float64.


## Running the tests

```bash
# From the repo root
python -m pytest tests/ -v -s

# Correctness only (fast)
python -m pytest tests/ -v -s -k "not Latency"

# Single class
python -m pytest tests/test_tensorstats.py::TestGrid -v -s
```

All tests go through `ts.StatsComputer` — the sole public interface. The suite
covers correctness, uint8 histogram path, stride accuracy, grid shapes and values,
higher-dimensional tensors (2D through 6D), and latency benchmarks.


## Performance

**Linux** (GCC -O3 -march=native, `axes=[None,(0,1)], grid=(4,4,2)`):

| Shape | dtype | stride | `sc.compute()` p50 |
|-------|-------|--------|--------------------|
| 64×64×3 | uint8 | 1 | 0.115ms |
| 64×64×3 | uint8 | (4,4,1) | 0.118ms |
| 64×64×3 | float64 | 1 | 0.120ms |
| 64×64×3 | float64 | (4,4,1) | 0.059ms |
| 128×128×3 | uint8 | 1 | 0.414ms |
| 128×128×3 | float64 | (4,4,1) | 0.202ms |

**Windows** (MSVC /O2 /arch:AVX2) is typically 2–3× slower than Linux on the
grid path due to MSVC generating less efficient scatter loops. The uint8
histogram path and axes reductions are closer to parity.

`StatsComputer` is ~1.4–2× faster than constructing fresh each call (larger
arrays benefit more), because the grid path retains its precomputed cell-index
array and output buffer.


## Implementation notes

See [`NOTES.md`](NOTES.md) for a detailed record of
what was tried, what worked, and the reasoning behind the current design — useful
if you're revisiting performance or considering further changes.

**uint8 histogram path** (global and per-channel axes reductions):
Builds `hist[256]` using 4-way parallel counters, then computes all 4 moments
with 256 FMA operations. Faster than the float two-pass for N ≥ 256 pixels because
it avoids N float reads in pass 2 — only 256 histogram bins are touched instead.

**Grid scatter path** (`_GridStatsComputerImpl`, internal):
Each grid level precomputes a `cell_of[]` array (int16, one entry per pixel)
mapping each pixel flat index to its cell in that level. The scatter loops then do
one table lookup per pixel instead of `ndim` multiplications and additions. For a
pyramid the sweep is pixel-outer, level-inner: every pixel is read once and
scattered into all levels, so the tensor is read once (all-histogram) or twice
(any direct level) regardless of how many levels are requested. Direct levels do
two passes (sum into `mu[]`, then subtract mean and accumulate moments); uint8
levels with large cells use a per-cell 256-bin histogram instead. The cell-index
arrays and all accumulator vectors are retained across `compute()` calls — no heap
allocation per frame.

**Stride**: applied in the axes path by stepping the flat index or per-axis
coordinate, and in the grid path by visiting a precomputed list of sampled flat
indices (built once at config). Grid cell boundaries stay at full resolution and
non-visited pixels are skipped, so each cell reports the exact moments of its
visited pixels. Empty cells (when stride, or `n_cells > shape`, leaves a cell
unvisited) are written as zeros.

**Output**: axes reductions allocate a fresh `double[]` and wrap it in a nanobind
capsule. Grid reductions write into a retained `out_buf_` and Python immediately
copies it before returning (safe for consecutive calls on the same `StatsComputer`).
