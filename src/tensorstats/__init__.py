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
      stride=(4, 4, 1),
      grid=(4, 4, 2),
  )
  result = sc.compute(hsv_frame)   # reuses internal buffers every call
  result["global"]   # (4,)           exact global moments
  result["0,1"]      # (3, 4)         exact per-channel moments
  result["grid"]     # (16,16,4,4)    exact grid moments
"""

from __future__ import annotations
import numpy as np

from . import tensorstats_core as _core

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


def _parse_grid(grid, ndim: int) -> list[int]:
    """grid[d]=k → 2^k cells along axis d. None or empty → no grid."""
    if grid is None:
        return []
    if isinstance(grid, int):
        return [grid] * ndim
    g = list(grid)
    if len(g) != ndim:
        raise ValueError(f"grid length {len(g)} must match ndim={ndim}")
    if any(v < 0 for v in g):
        raise ValueError("grid values must be >= 0")
    return g


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
    grid      : None=no grid, int k (2^k cells/axis), or per-axis tuple e.g. (4,4,2).
    n_moments : 1–4 (default 4).

    Result keys and shapes
    ----------------------
    "global"        (n_moments,)              when axes includes None
    "<a>,<b>,..."   (*kept_shape, n_moments)  for each axis-tuple in axes
    "grid"          (*cell_shape, n_moments)  when grid is set

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
        self._grid_vec = _parse_grid(grid, self._ndim)
        self._has_grid = bool(self._grid_vec)
        self._gsc: _GridStatsComputerImpl | None = None
        if self._has_grid:
            self._gsc = _GridStatsComputerImpl()
            self._gsc.set_config(list(self._shape), self._grid_vec, self._n_moments)

    def compute(self, arr: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute exact central moments for arr.

        arr must match the shape passed at construction.


        """
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

        # Grid reduction (retains buffers, returns copy — view lifetime is tied to
        # this StatsComputer object, so always copy for safety)
        if self._has_grid:
            if arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]:
                result["grid"] = self._gsc.compute_u8(arr).copy()
            else:
                result["grid"] = self._gsc.compute_f64(
                    np.ascontiguousarray(arr, np.float64)
                ).copy()

        return result


__all__ = ["StatsComputer"]
