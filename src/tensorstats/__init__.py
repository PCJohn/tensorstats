"""
tensorstats — fast two-pass central moment computation.

Output convention: moments-LAST.
All output arrays have shape (*reduction_output_shape, n_moments):
  "global"  → (n_moments,)
  "0,1"     → (C, n_moments)
  "grid"    → (*cell_shape, n_moments)

Moments layout (last axis):
  [..., 0]  mean
  [..., 1]  2nd central moment (variance)
  [..., 2]  3rd central moment
  [..., 3]  4th central moment

Derived standardised moments:
  std      = sqrt(m[..., 1])
  skewness = m[..., 2] / std**3
  kurtosis = m[..., 3] / m[..., 1]**2
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union

from .tensorstats_core import compute_f64, compute_f32, compute_u8

_AxisSpec = Optional[Union[int, tuple, list]]


def _normalise_one(spec, ndim: int) -> list[int]:
    if spec is None:
        return []
    if isinstance(spec, (int, np.integer)):
        spec = [int(spec)]
    result = []
    for a in spec:
        na = int(a) if int(a) >= 0 else int(a) + ndim
        if not (0 <= na < ndim):
            raise ValueError(f"axis {a} out of range for ndim={ndim}")
        result.append(na)
    return sorted(set(result))


def _parse_axes(axes, ndim: int) -> list[list[int]]:
    if axes is None:
        return [[]]
    if isinstance(axes, (int, np.integer)):
        return [_normalise_one(int(axes), ndim)]
    if isinstance(axes, tuple):
        return [_normalise_one(axes, ndim)]
    if isinstance(axes, list):
        if len(axes) == 0:
            return [[]]
        if isinstance(axes[0], (int, np.integer)):
            return [_normalise_one(axes, ndim)]
        return [_normalise_one(s, ndim) for s in axes]
    raise TypeError(f"unsupported axes type: {type(axes)}")


def _parse_grid(grid, ndim: int) -> list[int]:
    """
    Normalise the grid parameter to a list of ints of length ndim.
    grid[d] = number of power-of-2 subdivisions along axis d.
      0  → 1 cell (no subdivision)
      1  → 2 cells
      2  → 4 cells
      k  → 2^k cells
    Returns empty list if grid is None (no grid requested).
    """
    if grid is None:
        return []
    if isinstance(grid, int):
        return [grid] * ndim
    g = list(grid)
    if len(g) != ndim:
        raise ValueError(f"grid length {len(g)} must match arr.ndim={ndim}")
    if any(v < 0 for v in g):
        raise ValueError("grid values must be >= 0")
    return g


def compute(
    arr: np.ndarray,
    axes=None,
    stride=None,
    n_moments: int = 4,
    grid=None,
) -> dict[str, np.ndarray]:
    """
    Compute the first n_moments central moments of arr.

    Parameters
    ----------
    arr :
        Input array. uint8, float32, float64 accepted natively.
        Other dtypes cast to float64.
    axes :
        Axis-sets to reduce over (one or a list):
          None           → global (all axes)
          int            → single axis
          (int, ...)     → joint reduction
          [spec, ...]    → multiple reductions in one call
    stride :
        Subsample without malloc/resize.
          None or 1      → all elements
          int            → uniform flat step
          (int, ...)     → per-axis step
    n_moments :
        Number of moments (1–4, default 4).
    grid :
        Power-of-2 grid subdivision per axis.
          None           → no grid
          int k          → 2^k cells on every axis
          (k0, k1, ...)  → 2^k0 cells on axis 0, 2^k1 on axis 1, etc.
        Result stored in result["grid"] with shape (*cell_shape, n_moments).
        uint8 histogram path used for cells >= 256 pixels;
        direct loop used for smaller cells.

    Returns
    -------
    dict[str, np.ndarray]  — moments-LAST convention
        "global"  → (n_moments,)
        "0,1"     → (C, n_moments)
        "0"       → (W, C, n_moments)
        "grid"    → (*cell_shape, n_moments)
    """
    ndim = arr.ndim

    # Stride
    if stride is None or stride == 1:
        stride_vec = [1] * ndim
    elif isinstance(stride, int):
        stride_vec = [stride] * ndim
    else:
        stride_vec = list(stride)
        if len(stride_vec) != ndim:
            raise ValueError(f"stride length must match arr.ndim={ndim}")

    axes_list = _parse_axes(axes, ndim)
    grid_vec  = _parse_grid(grid, ndim)

    if arr.dtype == np.uint8 and arr.flags['C_CONTIGUOUS']:
        return compute_u8(arr, axes_list, stride_vec, n_moments, grid_vec)
    if arr.dtype == np.float32 and arr.flags['C_CONTIGUOUS']:
        return compute_f32(arr, axes_list, stride_vec, n_moments, grid_vec)
    arr64 = np.ascontiguousarray(arr, dtype=np.float64)
    return compute_f64(arr64, axes_list, stride_vec, n_moments, grid_vec)


__all__ = ["compute"]
