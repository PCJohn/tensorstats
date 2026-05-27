# tensorstats — Development Notes

These notes record the algorithmic and systems decisions behind tensorstats,
including what was tried, what worked, and why certain approaches were abandoned.
Intended for anyone revisiting performance or considering further changes.

---

## Benchmark: final latency numbers

`StatsComputer.compute()` p50 latency, `axes=[None,(0,1)], grid=(4,4,2)`.

**Linux** (GCC -O3 -march=native -ffast-math):

| Shape | dtype | stride | p50 |
|-------|-------|--------|-----|
| 64×64×3 | uint8 | 1 | 0.115ms |
| 64×64×3 | uint8 | (4,4,1) | 0.118ms |
| 64×64×3 | float64 | 1 | 0.120ms |
| 64×64×3 | float64 | (4,4,1) | 0.059ms |
| 128×128×3 | uint8 | 1 | 0.414ms |
| 128×128×3 | uint8 | (4,4,1) | 0.460ms |
| 128×128×3 | float64 | (4,4,1) | 0.202ms |

**Windows** (MSVC /O2 /arch:AVX2 /fp:fast):

| Shape | dtype | stride | p50 |
|-------|-------|--------|-----|
| 64×64×3 | uint8 | 1 | ~0.176ms |
| 64×64×3 | uint8 | (4,4,1) | ~0.190ms |
| 64×64×3 | float64 | 1 | ~0.350ms |
| 64×64×3 | float64 | (4,4,1) | ~0.180ms |
| 128×128×3 | uint8 | 1 | ~0.650ms |
| 128×128×3 | float64 | (4,4,1) | ~0.550ms |

**Baseline for comparison:** pure NumPy computing the same quantities
(global mean/var, per-channel, no grid) on 64×64×3 uint8: ~0.860ms on Windows.
The grid adds significant cost over NumPy's per-cell approach at large cell
counts — tensorstats wins because it avoids per-cell Python overhead entirely.

---

## What was built: algorithmic approaches

### 1. uint8 histogram path (global + per-channel)

**Idea:** for uint8 data, values are bounded to [0, 255]. Build `hist[256]` with
integer increments, then compute all 4 moments with 256 float FMAs.
This is O(256) in the moment computation regardless of input size N.

**Why it wins:** the two-pass float approach reads N doubles twice (sum pass,
moments pass). For a 64×64×3 array, N=12,288. After the histogram, the second
pass is always 256 iterations — 48× fewer. The crossover is at ~256
pixels/cell (`HIST_THRESHOLD`).

**4-way parallel counters:** `build_hist4` accumulates into four separate
histograms `h0..h3`, then sums them. This breaks the loop-carried dependency
on `hist[v]++`, letting the CPU pipeline the increments.

**Measured speedup:** 6–7× vs NumPy global uint8 on 64×64×3.

### 2. Grid scatter with precomputed cell index

**Idea:** instead of computing each pixel's cell index from coordinates on every
call, precompute a flat `cell_of[]` array (int16) at construction time. The
scatter loop becomes `mu[cell_of[i]] += data[i]` — one table lookup per pixel.

**Why it matters:** without this, computing the cell index requires `ndim`
multiplications and additions per pixel. With it, it's one int16 load and one
addition. The `cell_of[]` array (24KB for 64×64×3) fits in L1 cache and stays
hot across consecutive calls.

**Output buffer retention:** `_GridStatsComputerImpl` owns all accumulator
vectors (`mu_`, `m2_`, `m3_`, `m4_`, `counts_`) and output buffer (`out_buf_`).
These are zero-filled each call instead of reallocated. The C++ object is
constructed once per `StatsComputer` and lives for its lifetime.

**Measured speedup vs fresh-allocated grid:** ~2× at 64×64×3, ~2× at 128×128×3.

### 3. Pixel-outer scatter (vs cell-outer iteration)

**Earlier approach (cell-outer):** for each of N_cells cells, collect the flat
indices of pixels in that cell, then compute moments on that subset. For a
16×16×4=1024-cell grid on 64×64×3, this meant 1024 vector setups and boundary
checks.

**Current approach (pixel-outer):** one linear pass over all N pixels. Each
pixel looks up its cell via `cell_of[i]` and scatters into the accumulator.
A second pass then finalises moments for each cell. No per-cell setup overhead.

**Why the earlier approach was slow:** 1024 cells × 12 pixels/cell = 12,288
pixel-ops, but also 1024 vector clears and boundary-check loops. The scatter
approach has the same pixel-op count but zero per-cell overhead.

---

## Summary: what worked

| Technique | Measured gain | Where in code |
|-----------|---------------|---------------|
| uint8 histogram path | 6–7× vs NumPy global | `global_u8_hist`, `last_axis_u8_hist` |
| Pixel-outer scatter vs cell-outer | ~5.8× on grid path | `_GridStatsComputerImpl` |
| Retained output buffer (no alloc per call) | ~2× overall grid path | `_GridStatsComputerImpl` |
| Precomputed `cell_of[]` int16 array | ~1.1× on scatter loops alone | `_GridStatsComputerImpl` |
| 4-way parallel histogram counters | ~1.1–1.2× | `build_hist4` |

---

## What did NOT work (and why)

### Pébay single-pass algorithm for moments

**What:** the Welford/Pébay online algorithm computes all 4 moments in a single
pass without needing the mean first.

**Why it failed:** 0.60× — measurably slower. The algorithm does ~6 dependent
float ops per pixel (delta, delta_n, term1, then M2, M3, M4 updates with
cross-terms). The two-pass approach does 1 addition in pass 1 and 3
multiplications in pass 2, with better instruction-level parallelism. The cache
benefit of one pass is outweighed by the heavier per-pixel work.

### Pre-allocated accumulator buffers (stateful vectors)

**What:** retain `sums_`, `counts_`, etc. as class members and zero-fill each
call, rather than allocating fresh.

**Why it did not help:** modern allocators cache small allocations — malloc/free
for 8KB is nearly free after warmup. `std::fill` on hot-cache vectors costs as
much as a fresh zero-initialized allocation from the cache. The real win was
the *output buffer* retention (eliminating nanobind capsule + Python object
construction), not the accumulator buffers themselves.

### Reducing output to derived moments in C++

**What:** instead of returning raw `[mean, var, m3, m4]`, compute and return
`[mean, std, skewness, kurtosis]` directly in C++, saving numpy post-processing
on the Python side.

**Why it was wrong:** numpy's per-call overhead (~0.003ms) on small (16,16)
arrays adds up across 3 channels × 4 ufunc calls, so the saving looked real.
But this baked application-level decisions (which derived quantities to compute)
into the library. It also tempted further shortcuts — specifically, aggregating
spatial cell values to produce global scalars, which is mathematically wrong
(mean of cell means ≠ global mean; variance cannot be recovered at all from
averaged cell variances). Reverted. The correct boundary: library returns exact
raw moments; callers derive what they need.

### Deriving global stats from grid cell spatial averages

**What:** eliminate the separate `axes=[None,(0,1)]` C++ call by averaging the
16×16 grid cell values to approximate global and per-channel scalars.

**Why it was wrong:** mean of cell means ≠ global mean when cells have unequal
counts (which they do with any non-trivial grid). Variance requires the parallel
variance formula (within-group + between-group). Skewness and kurtosis cannot be
aggregated from cell values at all. Measured error was ~7% on variance, higher on
higher moments. Reverted completely.

### Spatial stride inside `_GridStatsComputerImpl`

**What:** add a `spatial_stride` parameter so the grid computer only visits every
Nth pixel, reducing scatter work proportionally.

**Why it was reverted:** this is an accuracy tradeoff that belongs in the
application, not the library. The library computes exact moments over whatever
array it receives. If a caller wants approximate grid stats from a subsampled
input, they should pass a pre-subsampled array and document the tradeoff
explicitly. Keeping this in the library conflated two different responsibilities.

### `StatsComputer.compute()` returning a view for grid output

**What:** return `result["grid"]` as a zero-copy numpy view into the retained
C++ buffer, saving one 32KB copy per call.

**Why it was wrong:** when `StatsComputer` is constructed as a temporary
(`ts.StatsComputer(shape=...).compute(arr)`), the C++ object is garbage-collected
before Python reads the view, returning corrupted data. The view lifetime is
silently tied to the C++ object lifetime, which is not safe to assume.
Changed to always copy — costs ~0.001ms but is unconditionally safe.

---

## Platform-specific findings

### MSVC vs GCC/Clang scatter loop performance

The grid scatter loops (`mu[cell_of[i]] += data[i]`) are ~2.8× slower on MSVC
than GCC at the same input size. GCC auto-vectorizes these scatter-accumulate
loops with `-O3 -march=native`. MSVC does not.

`#pragma loop(ivdep)` is added before the scatter loops in `core.cpp`. This
tells MSVC there are no loop-carried dependencies between iterations (different
pixels always write to potentially different cells — MSVC cannot prove this
statically). This enables partial vectorization but does not fully close the gap.

**Practical implication:** for 64×64×3 uint8 with `axes=[None,(0,1)], grid=(4,4,2)`,
Windows p50 is ~0.176ms vs ~0.115ms on Linux. The uint8 histogram path and
per-axis reductions without a grid are closer to parity (1.5–2× slower on Windows
vs Linux).

### Windows `time.perf_counter` timer floor

Single `time.perf_counter()` pair measurements on Windows cannot reliably resolve
below ~0.1ms for fast operations. For profiling sub-0.1ms operations on Windows,
use batch timing: run N iterations in one timed block and divide. This is what the
unit tests do in `timeit_ms()` — each timed chunk runs for ~20ms wall time so
scheduling noise is negligible relative to the measurement.

### Mac ARM (Apple Silicon)

NumPy's float64 global reduction is faster than tensorstats at 64×64×3 and
larger on Mac ARM, because numpy is backed by Apple's Accelerate framework (NEON
SIMD, highly optimized single-pass reductions). tensorstats wins at ≤32×32 (where
Python call overhead dominates) and for uint8 (histogram path). This is a
hardware-platform reality, not a bug — the test suite asserts only where wins are
reliable across all platforms.

---

## Nanobind overhead

Calling a C++ function via nanobind costs ~0.003ms on Linux (measured by calling
`StatsComputer.compute()` on a trivially small 2×2×1 array where all C++ work is
negligible). This is the Python→C++ boundary crossing cost.

For grid output, the main overhead before the retained buffer approach was not the
computation but constructing the return value: `new double[]` + `nb::capsule` +
`nb::ndarray` + dict insertion. For `grid=(4,4,2)` with 1024 cells, this
construction cost was ~0.073ms. With the retained `out_buf_` and a Python-side
copy, it is ~0.001ms.

**General principle:** for nanobind extensions with fast computations and small
outputs called repeatedly, Python object creation can dominate. Solutions: retain
output buffers (as done here for the grid), or batch multiple outputs into a
single return value.

---

## Design principles

**Library returns raw moments; callers derive.** std, skewness, kurtosis are
derived quantities. Callers have different needs — some want std, some want
kurtosis, some want neither. The library should not make this choice.

**Accuracy is non-negotiable.** Several speed attempts introduced error. All were
reverted. Accuracy tradeoffs belong in the application where they can be
explicitly validated and documented.

**Profile before optimizing.** Several expected wins (Pébay single-pass,
pre-allocated accumulators) turned out to be slower or irrelevant. The actual
wins were often different from the expected ones.

**Separation of concerns.** The grid path (`_GridStatsComputerImpl`) and the axes
path (free functions in `compute_typed`) have different memory management needs
and are correctly implemented separately. Merging them would not have simplified
the code or improved performance.

**The test suite must be honest.** Early tests called `ts.compute()` (a free
function wrapper) and were blind to bugs in `StatsComputer` itself, including
accuracy corruption that went undetected for several iterations. All tests now go
through `StatsComputer` with correctness assertions against NumPy reference
implementations.
