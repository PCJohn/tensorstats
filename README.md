# tensorstats

Fast central moment computation for tensors and arrays, implemented in C++
with a Python/NumPy interface via [nanobind](https://github.com/wjakob/nanobind).

Designed as a high-speed gate before expensive vision pipelines — blank frame
detection, scene change detection, image statistics for classifier routing.

---

## Algorithms

### General case (float32, float64)

Two numerically stable passes over the data. No temporary arrays allocated.

**Pass 1 — mean:**
```
mu = Σ x_i / n
```

**Pass 2 — central moments:**
```
for each x_i:
    d  = x_i - mu
    d2 = d * d
    m2 += d2          # variance accumulator
    m3 += d2 * d
    m4 += d2 * d2
```

This avoids the catastrophic cancellation of the raw-moment approach
(`Σx²/n − μ²`), which loses precision for near-constant arrays.

### uint8 special case — histogram path

uint8 pixels take values in [0, 255]. Instead of iterating over N pixels
for the moment accumulation, tensorstats builds a 256-bin histogram in one
pass and computes moments from the histogram in a fixed 256 iterations.

```
Pass 1: hist[v]++ for each pixel v           # N integer increments
Pass 2: mu = Σ hist[v]*v / n                 # exact integer mean
Pass 3: m2,m3,m4 = Σ hist[v]*(v-mu)^k       # only 256 float FMAs
```

The histogram build uses 4-way parallel counters (h0/h1/h2/h3 merged at
the end) to reduce write-after-write stalls on hot bins. The mean is
computed from an exact integer sum — no floating-point rounding in pass 1.
This gives the same numerical stability as the two-pass float approach.

**Why this is faster:** at 64×64×3 (12,288 pixels), the moment pass does
12,288 float FMAs in the general case vs 256 FMAs from the histogram.
The gains grow with array size.

### Three inner-loop paths (selected at runtime)

| Path | When used | Key property |
|---|---|---|
| **Global** | `axes=None` | Single accumulator, straight loop, fully AVX2-vectorized |
| **LastAxis** | e.g. `axes=(0,1)` on HxWxC | Stride-C column loops — no modulo, AVX2-vectorized |
| **General** | Arbitrary axes | Precomputed `(flat_index, bucket)` pairs |

For uint8, Global and LastAxis use the histogram path. General uses the
direct loop (histogram doesn't simplify arbitrary reductions).

---

## Performance

Measured on two machines. All times in milliseconds.

### Linux / GCC (flags: `-O3 -march=native -ffast-math`)

**Global moments — `ts.compute(arr, axes=None)`**

| shape | numpy | ts float64 | ts uint8 | f64 speedup | u8 speedup |
|---|---|---|---|---|---|
| 16×16×3 | 0.014 | 0.004 | 0.004 | 3.5× | 3.4× |
| 32×32×3 | 0.020 | 0.009 | 0.005 | 2.3× | 4.2× |
| 64×64×3 | 0.043 | 0.029 | 0.011 | 1.5× | 4.0× |
| 128×128×3 | 0.586 | 0.105 | 0.020 | 5.6× | 29× |
| 256×256×3 | 3.133 | 0.432 | 0.068 | 7.3× | **46×** |

**Global + per-channel — `ts.compute(arr, axes=[None, (0,1)])`**

| shape | numpy | ts float64 | ts uint8 | f64 speedup | u8 speedup |
|---|---|---|---|---|---|
| 16×16×3 | 0.041 | 0.009 | 0.011 | 4.7× | 3.9× |
| 32×32×3 | 0.052 | 0.018 | 0.014 | 2.8× | 3.8× |
| 64×64×3 | 0.087 | 0.059 | 0.024 | 1.5× | 3.6× |
| 128×128×3 | 0.259 | 0.213 | 0.051 | 1.2× | 5.1× |
| 256×256×3 | 3.403 | 0.918 | 0.177 | 3.7× | **19×** |

### Windows / MSVC (flags: `/O2 /arch:AVX2 /fp:fast`)

**Global moments — `ts.compute(arr, axes=None)`**

| shape | numpy | ts float64 | ts uint8 | f64 speedup | u8 speedup |
|---|---|---|---|---|---|
| 16×16×3 | 0.016 | 0.004 | 0.003 | 4× | 5× |
| 32×32×3 | 0.022 | 0.006 | 0.003 | 4× | 7× |
| 64×64×3 | 0.035 | 0.019 | 0.006 | 1.8× | 6× |
| 128×128×3 | 0.111 | 0.057 | 0.018 | 2× | 6× |
| 256×256×3 | 2.244 | 0.232 | 0.059 | 10× | **34×** |

### Stride accuracy (gaussian blob, uint8, vs full-image numpy)

Stride subsamples the array in memory order without malloc or resize.
Error vs computing on all pixels:

| stride | mean err | variance err | m3 err | m4 err |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | ~1e-16 |
| 2 | ~1e-4 | ~1e-4 | ~1e-4 | ~1e-4 |
| 4 | ~1e-4 | ~1e-4 | ~1e-4 | ~1e-4 |
| 8 | ~2e-4 | ~7e-4 | ~7e-4 | ~1e-3 |

Stride ≤ 2 is recommended for gating/triggering use cases. Mean is the
most robust moment; m3/m4 degrade faster at higher strides.

---

## Install

`tensorstats` is a C++ extension built with CMake. `pip install -e .`
is the build step — it invokes CMake automatically via `scikit-build-core`.
First build takes 30–60 seconds.

### Common steps (all platforms)

```bash
git clone https://github.com/PCJohn/tensorstats
cd tensorstats
pip install scikit-build-core nanobind
pip install -e .
python -c "import tensorstats; print('ok')"
```

### Platform prerequisites

**Linux (Ubuntu/Debian)**
```bash
sudo apt install build-essential cmake python3-dev
```

**macOS**
```bash
xcode-select --install   # Xcode command-line tools (clang + make)
brew install cmake
```

**Windows**

1. Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   — select the **"Desktop development with C++"** workload (installs MSVC + Windows SDK)
2. Install [CMake](https://cmake.org/download/) — check "Add CMake to system PATH"
3. **Open "Developer Command Prompt for VS 2022"** from the Start menu.
   Regular PowerShell/CMD will fail — CMake needs the VS environment to find the compiler.
4. Run the common steps above from that prompt.

---

## API

### `ts.compute(arr, axes=None, stride=None, n_moments=4)`

Compute the first `n_moments` central moments of `arr` for each requested
axis-set.

**Parameters**

| param | type | description |
|---|---|---|
| `arr` | `np.ndarray` | Input array. uint8, float32, float64 accepted natively (no copy). Other dtypes cast to float64. |
| `axes` | see below | Which axis-sets to reduce over. |
| `stride` | `int` or `tuple[int,...]` | Subsample the array — skip elements without malloc or resize. `None` or `1` = use all elements. Scalar applies to all axes; tuple applies per-axis. |
| `n_moments` | `int` | Number of moments to compute (1–4, default 4). |

**`axes` argument**

| value | meaning |
|---|---|
| `None` | global — reduce over all axes |
| `int` | reduce over that single axis |
| `(int, int, ...)` | reduce over those axes jointly |
| `[spec, spec, ...]` | compute multiple reductions in one call |

**Returns**

`dict[str, np.ndarray]` — one entry per requested reduction.

| key | output shape | example |
|---|---|---|
| `"global"` | `(n_moments,)` | all-axes reduction |
| `"0,1"` | `(n_moments, C)` | axes (0,1) reduced — per-channel for HxWxC |
| `"0"` | `(n_moments, W, C)` | axis 0 reduced |

**Moments layout (axis 0 of each output array)**

| index | meaning | standardised form |
|---|---|---|
| 0 | mean | — |
| 1 | 2nd central moment (variance) | — |
| 2 | 3rd central moment | skewness = `m[2] / sqrt(m[1])**3` |
| 3 | 4th central moment | kurtosis = `m[3] / m[1]**2` |

---

## Usage examples

```python
import numpy as np
import tensorstats as ts

img = cv2.imread("frame.jpg")   # (H, W, 3) uint8 — accepted natively

# --- Single reduction ---
result = ts.compute(img, axes=None)
result["global"]      # shape (4,): [mean, variance, m3, m4] over all pixels

# --- Multiple reductions in one call ---
result = ts.compute(img, axes=[None, (0, 1), 0])
result["global"]      # shape (4,)    — global stats
result["0,1"]         # shape (4, 3)  — per-channel stats
result["0"]           # shape (4, W, 3) — per-column stats

# --- Derived standardised moments ---
m = result["global"]
std      = np.sqrt(m[1])
skewness = m[2] / std**3
kurtosis = m[3] / m[1]**2

# --- Stride: subsample without malloc/resize ---
# scalar stride: flat memory step  data[0], data[s], data[2s], ...
ts.compute(img, axes=None, stride=2)         # ~1.5× faster, small error

# tuple stride: per-axis  — skip rows/cols, keep all channels
ts.compute(img, axes=None, stride=(2, 2, 1))

# --- Fewer moments (faster if you only need mean + variance) ---
ts.compute(img, axes=None, n_moments=2)      # returns shape (2,)
```

**Note on stride semantics:** scalar `stride=s` steps through the flat
C-contiguous array (`data[0], data[s], data[2s], ...`). This is not the
same as `arr[::s, ::s, ::s]` (per-axis numpy slicing). Use a tuple to
stride specific axes independently.

**Note on resize/interpolation:** when passing a downsampled thumbnail,
the interpolation method affects which features survive:
- `cv2.INTER_NEAREST` — no blending, preserves sharp edges (good for feature-based stats)
- `cv2.INTER_AREA` — anti-aliased (good for photometric stats)

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v -s
```

Tests cover: correctness vs numpy reference, numerical stability
(near-constant, large values, flat images, gradients), stride accuracy,
native dtype handling (uint8/float32), all image edge cases
(all-zeros, all-255, gaussian blob, checkerboard, two-tone), and latency
benchmarks asserting tensorstats beats numpy.

---

## License

Apache 2.0
