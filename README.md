# tensorstats

Fast single-pass central moment computation for tensors/arrays, implemented
in C++ with a Python/NumPy interface via nanobind.

## Why

NumPy's moment computation requires multiple passes and allocates large
temporary arrays. For small arrays (e.g. 64×64 image thumbnails) the
allocation overhead dominates. `tensorstats` avoids temporaries:

- **Pass 1**: mean per bucket — pure sum, SIMD-vectorized (AVX2)
- **Pass 2**: `d=x-mu`, accumulate `d²`, `d³`, `d⁴` — 4 FMAs/element, AVX2

Typical speedups vs NumPy (global moments, float64):

| Shape | speedup |
|---|---|
| 16×16×3 | ~5× |
| 32×32×3 | ~2.7× |
| 64×64×3 | ~1.8× |
| 128×128×3 | ~1.6× |
| 256×256×3 | ~2× |

## Install

```bash
pip install -e .
```

### Prerequisites by platform

**Linux**
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake python3-dev
pip install scikit-build-core nanobind
pip install -e .
```

**macOS**
```bash
# Xcode command-line tools
xcode-select --install
brew install cmake
pip install scikit-build-core nanobind
pip install -e .
```

**Windows**

1. Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   — select **"Desktop development with C++"** workload
2. Install [CMake](https://cmake.org/download/) — add to PATH during install
3. Open **"Developer Command Prompt for VS 2022"** (not a regular terminal)
4. Run:
```bat
pip install scikit-build-core nanobind
pip install -e .
```

> **Note**: must use the VS Developer Command Prompt so CMake finds the MSVC compiler.
> PowerShell/CMD without VS environment will fail with "compiler not found".

## Usage

```python
import numpy as np
import tensorstats as ts

# Any numpy array or cv2 image (uint8/float32/float64)
img = cv2.imread("frame.jpg")   # (H, W, 3) uint8

# Multiple reductions in one call
result = ts.compute(img, axes=[
    None,     # global  → shape (4,)
    (0, 1),   # per-channel → shape (4, 3)
    0,        # per-column  → shape (4, W, C)
])

result["global"]      # [mean, var, m3, m4] over all pixels
result["0,1"]         # [mean, var, m3, m4] per channel
result["0,1"][0]      # per-channel means
result["0,1"][1]      # per-channel variances

# Derived standardised moments
m = result["global"]
std      = np.sqrt(m[1])
skewness = m[2] / std**3
kurtosis = m[3] / m[1]**2

# Stride: subsample without resize/malloc (~2× faster at stride=2)
result = ts.compute(img, axes=None, stride=2)
result = ts.compute(img, axes=[None, (0,1)], stride=(2, 2, 1))
```

## Output format

`compute()` returns a `dict` mapping axis-key → `ndarray` of shape
`(n_moments, *output_shape)`:

| key | meaning | shape for (H,W,C) input |
|---|---|---|
| `"global"` | all axes | `(n_moments,)` |
| `"0,1"` | axes (0,1) reduced | `(n_moments, C)` |
| `"0"` | axis 0 reduced | `(n_moments, W, C)` |

Moments layout:

| index | meaning |
|---|---|
| 0 | mean |
| 1 | 2nd central moment (variance) |
| 2 | 3rd central moment |
| 3 | 4th central moment |

## Performance tips

### Use float32 input
float32 uses 2× wider AVX2 lanes (8 floats vs 4 doubles) → ~2× faster:
```python
img_f32 = img.astype(np.float32)
result = ts.compute(img_f32, axes=None)   # ~2× faster than float64
```

### Use stride for approximate but fast statistics
`stride=2` halves the elements → ~2× faster, small error for natural images:
```python
# stride=1: exact. stride=2: ~1% mean error, ~10% variance error (natural images)
result = ts.compute(img, axes=None, stride=2)
```

Accuracy degrades gracefully with stride — mean is most robust, higher
moments (m3, m4) degrade faster. For gating/triggering use cases
stride ≤ 2 is recommended.

### Flat stride vs per-axis stride
`stride=2` (scalar) applies a **flat step** in memory order:
`data[0], data[2], data[4], ...`. This is NOT the same as `arr[::2,::2,::2]`
(which strides each axis independently). Use a tuple to stride specific axes:
```python
ts.compute(img, stride=(2, 2, 1))   # skip rows/cols, keep all channels
```

### ⚠️ Resize/interpolation scheme matters
When using `tensorstats` as a gate before a heavy vision pipeline,
the interpolation method used to produce the input thumbnail
**significantly** affects which features survive downsampling:

- `cv2.INTER_NEAREST` — no blending, preserves sharp edges and corners
- `cv2.INTER_AREA` — anti-aliased, better for photometric stats
- Stride subsampling (`arr[::s, ::s]`) — zero-copy numpy view, identical
  semantics to INTER_NEAREST at 0.5× but without a resize call

## Algorithm

Two-pass numerically stable algorithm:

**Pass 1** (mean):
```
sum = Σ x_i
mu  = sum / n
```

**Pass 2** (central moments, no temporaries):
```
for each x_i:
    d  = x_i - mu
    d2 = d * d
    m2 += d2
    m3 += d2 * d
    m4 += d2 * d2
```

Avoids the catastrophic cancellation of raw-power-sum methods
(`Σx²/n - mu²`) which fails for near-constant arrays.

Three specialised inner-loop paths — selected at runtime:
1. **Global** — straight loop, fully AVX2-vectorized
2. **LastAxis** — stride-C column loops (no modulo), e.g. per-channel for HxWxC
3. **General** — precomputed `(flat_index, bucket)` pairs, arbitrary axes + strides

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v -s
```

Tests cover correctness (vs numpy reference), numerical stability
(near-constant arrays, large values), stride accuracy, native dtype
handling, and latency benchmarks.

## License

Apache 2.0
