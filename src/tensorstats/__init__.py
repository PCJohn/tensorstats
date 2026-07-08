"""
tensorstats — fast exact central moment computation.

Output convention: moments-LAST — all arrays have shape (*reduction_shape, n_moments).

Moments layout:
  [..., 0]  mean
  [..., 1]  variance
  [..., 2]  3rd central moment
  [..., 3]  4th central moment

Derive std/skewness/kurtosis from the raw moments:
  std      = sqrt(result[..., 1])
  skewness = result[..., 2] / std**3
  kurtosis = result[..., 3] / result[..., 1]**2

Usage:
  sc = ts.StatsComputer(
      shape=(64, 64, 3),
      axes=[None, (0, 1)],
      stride=(4, 4, 1),       # subsamples the grid too (subsampled-exact)
      grid=[(4, 4, 2), (5, 5, 0)],   # a grid pyramid: K levels in one pass
  )
  result = sc.compute(hsv_frame)   # reuses internal buffers every call
  result["global"]   # (4,)           exact global moments
  result["0,1"]      # (3, 4)         exact per-channel moments
  result["grid_0"]   # (16,16,4,4)    level 0 grid moments
  result["grid_1"]   # (32,32,1,4)    level 1 grid moments
"""

from __future__ import annotations
from typing import Any

import numpy as np

from . import tensorstats_core as _core  # type: ignore[attr-defined]

_compute_f64 = _core.compute_f64
_compute_f32 = _core.compute_f32
_compute_u8 = _core.compute_u8
_GridStatsComputerImpl = _core._GridStatsComputerImpl


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def _normalise_one(spec, ndim: int) -> list[int]:
    """Normalise one axis spec (None / int / tuple) → sorted list of ints."""
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
        if not axes:
            return [[]]
        if isinstance(axes[0], (int, np.integer)):
            return [_normalise_one(axes, ndim)]
        return [_normalise_one(s, ndim) for s in axes]
    raise TypeError(f"unsupported axes type: {type(axes)}")


def _parse_stride(stride, ndim: int) -> list[int]:
    if stride is None or stride == 1:
        return [1] * ndim
    if isinstance(stride, int):
        return [stride] * ndim
    sv = list(stride)
    if len(sv) != ndim:
        raise ValueError(f"stride length {len(sv)} must match ndim={ndim}")
    return sv


def _parse_grid(grid, ndim: int) -> list[list[int]]:
    """Parse grid into a list of per-axis exponent vectors (grid[d]=k -> 2^k cells).

    Accepts None (no grid), an int k, a single per-axis tuple/list, or a list of
    such specs (a grid pyramid). Returns [] when there is no grid. A single spec
    is the K=1 case of the same list-of-specs representation.
    """
    if grid is None:
        return []
    if isinstance(grid, int):
        return [[grid] * ndim]
    g = list(grid)
    if not g:
        return []
    specs = g if isinstance(g[0], (tuple, list)) else [g]
    out = []
    for s in specs:
        v = [s] * ndim if isinstance(s, int) else list(s)
        if len(v) != ndim:
            raise ValueError(f"grid spec length {len(v)} must match ndim={ndim}")
        if any(x < 0 for x in v):
            raise ValueError("grid values must be >= 0")
        out.append([int(x) for x in v])
    return out


# ---------------------------------------------------------------------------
# StatsComputer
# ---------------------------------------------------------------------------


class StatsComputer:
    """
    Stateful exact central moment computer for a fixed input shape and config.

    Construct once, call compute() on every frame/batch. The grid path retains
    its cell-index array and output buffer across calls (~2x faster than
    constructing fresh each time).

    Parameters
    ----------
    shape     : tuple — every compute() call must pass an array with this shape.
    axes      : None=global, int, tuple, or list of specs e.g. [None, (0,1)].
    stride    : None/1=no stride, int, or per-axis tuple e.g. (4,4,1).
    grid      : None=no grid, int k (2^k cells/axis), a per-axis tuple e.g.
                (4,4,2), or a LIST of such specs e.g. [(4,4,2),(5,5,0)] for a
                grid pyramid computed in a single pass. stride applies to the
                grid too (subsampled-exact moments).
    n_moments : 1–4 (default 4).

    Result keys and shapes
    ----------------------
    "global"        (n_moments,)              when axes includes None
    "<a>,<b>,..."   (*kept_shape, n_moments)  for each axis-tuple in axes
    "grid_<i>"      (*cell_shape, n_moments)  one per grid level, i = 0..K-1

    Supported dtypes: uint8, float32, float64. Other dtypes are cast to float64.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axes=None,
        stride=None,
        grid=None,
        n_moments: int = 4,
    ):
        self._shape = tuple(shape)
        self._ndim = len(shape)
        self._n_moments = n_moments
        self._axes_list = _parse_axes(axes, self._ndim)
        self._stride = _parse_stride(stride, self._ndim)
        self._grid_specs = _parse_grid(grid, self._ndim)
        self._has_grid = bool(self._grid_specs)
        self._gsc: Any = None  # opaque _GridStatsComputerImpl handle
        if self._has_grid:
            self._gsc = _GridStatsComputerImpl()
            self._gsc.set_config(
                list(self._shape),
                self._grid_specs,
                list(self._stride),
                self._n_moments,
            )

    def compute(self, arr: np.ndarray) -> dict[str, np.ndarray]:
        """Compute exact central moments for arr (must match the construction shape)."""
        if arr.shape != self._shape:
            raise ValueError(f"shape mismatch: expected {self._shape}, got {arr.shape}")

        # Axes reductions (global, per-channel, arbitrary)
        if arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]:
            result = _compute_u8(arr, self._axes_list, self._stride, self._n_moments)
        elif arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
            result = _compute_f32(arr, self._axes_list, self._stride, self._n_moments)
        else:
            result = _compute_f64(
                np.ascontiguousarray(arr, np.float64),
                self._axes_list,
                self._stride,
                self._n_moments,
            )

        # Grid reductions (K levels). Each level returns a VIEW into a retained
        # buffer whose lifetime is tied to this StatsComputer, so copy each one.
        if self._has_grid:
            if arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]:
                grids = self._gsc.compute_u8(arr)
            else:
                grids = self._gsc.compute_f64(np.ascontiguousarray(arr, np.float64))
            for i, gv in enumerate(grids):
                result[f"grid_{i}"] = gv.copy()

        return result


__all__ = ["StatsComputer"]
