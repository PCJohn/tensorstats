"""
tensorstats — fast central moment computation for tensors and arrays.

Moments-LAST output convention: all result arrays have shape (*reduction_shape, n_moments).

Moments layout (last axis):
  [..., 0]  mean
  [..., 1]  variance (2nd central moment)
  [..., 2]  3rd central moment
  [..., 3]  4th central moment

Derived quantities:
  std      = sqrt(result[..., 1])
  skewness = result[..., 2] / std**3
  kurtosis = result[..., 3] / result[..., 1]**2

Primary interface — stateful, retains buffers across calls (~2x faster on grid path):
  sc = ts.StatsComputer(shape=(64,64,3), axes=[None,(0,1)], stride=(4,4,1), grid=(4,4,2))
  result = sc.compute(arr)
  result["global"]   # (4,)        global moments
  result["0,1"]      # (3, 4)      per-channel moments (B,G,R or H,S,V)
  result["grid"]     # (16,16,4,4) spatial grid — VIEW into retained buffer

  NOTE: result["grid"] is a view into internal C++ memory. It is valid only until the
  next sc.compute() call. Use result["grid"].copy() to retain it longer.

Convenience wrapper — creates a temporary StatsComputer, copies grid output:
  result = ts.compute(arr, axes=[None,(0,1)], stride=(4,4,1), grid=(4,4,2))
"""

from __future__ import annotations
import numpy as np

from . import tensorstats_core as _core

_compute_f64       = _core.compute_f64
_compute_f32       = _core.compute_f32
_compute_u8        = _core.compute_u8
_GridComputerImpl  = _core._GridComputerImpl


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalise_one(spec, ndim: int) -> list[int]:
    """Normalise one axis spec (None / int / tuple) to a sorted list of ints."""
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
    """Return list of axis-lists for the C++ layer."""
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
    """grid[d] = k  →  2^k cells along axis d."""
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


def _dispatch_axes(arr: np.ndarray, axes_list, stride, n_moments) -> dict:
    """Call the typed C++ axes-reduction function for arr's dtype."""
    if arr.dtype == np.uint8 and arr.flags['C_CONTIGUOUS']:
        return _compute_u8(arr, axes_list, stride, n_moments)
    if arr.dtype == np.float32 and arr.flags['C_CONTIGUOUS']:
        return _compute_f32(arr, axes_list, stride, n_moments)
    return _compute_f64(np.ascontiguousarray(arr, np.float64),
                        axes_list, stride, n_moments)


# ---------------------------------------------------------------------------
# StatsComputer — stateful, retains buffers for fixed shape/config
# ---------------------------------------------------------------------------

class StatsComputer:
    """
    Stateful central moment computer for a fixed input shape and configuration.

    Construct once, call compute() on every frame. Retained buffers avoid
    per-call heap allocation. The grid path is ~2x faster than ts.compute()
    because the per-cell index array and output buffer are pre-allocated.

    Parameters
    ----------
    shape     : tuple — input shape, e.g. (64, 64, 3). Every compute() call
                must pass an array with exactly this shape.
    axes      : axis reduction spec. None=global, int, tuple, or list of specs.
                Examples: None, (0,1), [None, (0,1)].
    stride    : per-axis subsampling. None/1=no stride, int, or per-axis tuple.
    grid      : power-of-2 spatial grid. None=no grid, int k (2^k cells on
                every axis), or per-axis tuple e.g. (4,4,2).
    n_moments : number of moments to compute, 1–4 (default 4).

    Result keys
    -----------
    "global"   (n_moments,)          — when axes contains None
    "0,1"      (C, n_moments)        — when axes contains (0,1)
    "grid"     (*cell_shape,n_moments)— when grid is set; VIEW into retained buffer

    Example
    -------
    sc = ts.StatsComputer(
        shape=(64, 64, 3),
        axes=[None, (0, 1)],
        stride=(4, 4, 1),
        grid=(4, 4, 2),
    )
    result = sc.compute(hsv_frame)
    V_mean = float(result["global"][0])
    per_channel = result["0,1"]          # shape (3, 4)
    grid = result["grid"].copy()         # (16,16,4,4) — copy to keep past next call
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        axes=None,
        stride=None,
        grid=None,
        n_moments: int = 4,
    ):
        self._shape     = tuple(shape)
        self._ndim      = len(shape)
        self._n_moments = n_moments
        self._axes_list = _parse_axes(axes, self._ndim)
        self._stride    = _parse_stride(stride, self._ndim)
        self._grid_vec  = _parse_grid(grid, self._ndim)
        self._has_grid  = bool(self._grid_vec)
        self._gc: _GridComputerImpl | None = None
        if self._has_grid:
            self._gc = _GridComputerImpl()
            self._gc.set_config(list(self._shape), self._grid_vec, self._n_moments)

    def compute(self, arr: np.ndarray, _copy_grid: bool = False) -> dict[str, np.ndarray]:
        """
        Compute moments for arr.

        arr must have the shape passed at construction and be uint8, float32,
        or float64 (other dtypes are cast to float64).

        result["grid"] is a VIEW into a retained C++ buffer, valid until the
        next compute() call. Call .copy() if you need to keep it longer.
        """
        if arr.shape != self._shape:
            raise ValueError(f"shape mismatch: expected {self._shape}, got {arr.shape}")

        result = _dispatch_axes(arr, self._axes_list, self._stride, self._n_moments)

        if self._has_grid:
            if arr.dtype == np.uint8 and arr.flags['C_CONTIGUOUS']:
                grid_out = self._gc.compute_u8(arr)
            else:
                grid_out = self._gc.compute_f64(np.ascontiguousarray(arr, np.float64))
            result["grid"] = grid_out.copy() if _copy_grid else grid_out

        return result

    def set_shape(self, shape: tuple[int, ...]) -> None:
        """Change input shape and rebuild retained grid buffers."""
        self._shape = tuple(shape)
        self._ndim  = len(shape)
        if self._has_grid:
            self._gc.set_config(list(self._shape), self._grid_vec, self._n_moments)

    def set_grid(self, grid) -> None:
        """Change grid config and rebuild grid buffers. Axes path unaffected."""
        self._grid_vec = _parse_grid(grid, self._ndim)
        self._has_grid = bool(self._grid_vec)
        if self._has_grid:
            if self._gc is None:
                self._gc = _GridComputerImpl()
            self._gc.set_config(list(self._shape), self._grid_vec, self._n_moments)
        else:
            self._gc = None


# ---------------------------------------------------------------------------
# Convenience wrapper — stateless, one StatsComputer per call
# ---------------------------------------------------------------------------

def compute(
    arr: np.ndarray,
    axes=None,
    stride=None,
    n_moments: int = 4,
    grid=None,
) -> dict[str, np.ndarray]:
    """
    Compute central moments of arr (stateless convenience wrapper).

    For repeated calls on the same shape/config, use ts.StatsComputer directly —
    it retains internal buffers and is ~2x faster on the grid path.

    Parameters
    ----------
    arr       : uint8, float32, or float64 ndarray.
    axes      : None=global, int, tuple, or list of axis-specs.
    stride    : None=no stride, int, or per-axis tuple.
    n_moments : 1–4 (default 4).
    grid      : None, int k (2^k cells on every axis), or per-axis tuple.

    Returns
    -------
    dict[str, np.ndarray] with moments-LAST arrays.
    All outputs (including "grid") are independent copies — safe to keep.
    """
    return StatsComputer(
        arr.shape, axes=axes, stride=stride, grid=grid, n_moments=n_moments
    ).compute(arr, _copy_grid=True)


__all__ = ["StatsComputer", "compute"]
