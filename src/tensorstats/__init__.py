"""
tensorstats — fast two-pass central moment computation.

Moments layout in output arrays (axis 0):
    [0]  mean                  (1st raw moment)
    [1]  2nd central moment    (variance, not divided by σ)
    [2]  3rd central moment
    [3]  4th central moment

Standardised moments (skewness, kurtosis):
    std      = sqrt(m[1])
    skewness = m[2] / std**3
    kurtosis = m[3] / m[1]**2

Performance notes:
    • float32 input: 2× faster than float64 (AVX2 fits 8 floats vs 4 doubles)
    • stride=(2,...): halves elements → roughly halves compute time
    • For natural images stride ≤ 2 keeps mean/std error < 1%;
      higher moments degrade faster — stride ≤ 2 recommended for accuracy
    • The resize/interpolation scheme used before calling tensorstats
      meaningfully affects which features survive — see README
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Union

from .tensorstats_core import compute_f64, compute_f32, compute_u8

_AxisSpec = Optional[Union[int, Sequence[int]]]


def _normalise_one(spec: _AxisSpec, ndim: int) -> list[int]:
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
        first = axes[0]
        if isinstance(first, (int, np.integer)):
            return [_normalise_one(axes, ndim)]
        return [_normalise_one(s, ndim) for s in axes]
    raise TypeError(f"unsupported axes type: {type(axes)}")


def compute(
    arr: np.ndarray,
    axes=None,
    stride: Optional[Union[int, tuple[int, ...]]] = None,
    n_moments: int = 4,
) -> dict[str, np.ndarray]:
    """
    Compute the first ``n_moments`` central moments of ``arr`` along each
    requested axis-set in two passes over the data.

    Parameters
    ----------
    arr :
        Input array. Accepts uint8, float32, float64 natively (no copy made
        for float32/float64 C-contiguous arrays). Other dtypes are cast to
        float64. cv2 images (uint8 BGR) are accepted directly.
    axes :
        One axis-spec or a list of axis-specs:
          ``None``           → global (all axes)
          ``int``            → reduce over that axis
          ``(int, ...)``     → reduce over those axes jointly
          ``[spec, ...]``    → compute multiple reductions
        Example: ``axes=[None, (0,1), 2]``
    stride :
        Subsample the array without resize/malloc. Skips elements along
        each axis. Halving stride roughly halves compute time.
          ``None`` or ``1``  → use all elements (default)
          ``2``              → every other element on all axes
          ``(2, 2, 1)``      → stride 2 on axes 0,1; all elements on axis 2
        For natural images, stride ≤ 2 gives < 1% error on mean/std.
        Higher moments (skewness, kurtosis) degrade faster with stride.
    n_moments :
        Number of moments to compute (1–4, default 4).

    Returns
    -------
    dict  key → ndarray of shape ``(n_moments, *output_shape)``
        ``"global"``  — all-axes reduction
        ``"0,1"``     — axes=(0,1) reduction
        ``"2"``       — axis=2 reduction
    """
    ndim = arr.ndim

    # Normalise stride → list[int] of length ndim
    if stride is None or stride == 1:
        stride_vec = [1] * ndim
    elif isinstance(stride, int):
        stride_vec = [stride] * ndim
    else:
        stride_vec = list(stride)
        if len(stride_vec) != ndim:
            raise ValueError(f"stride length {len(stride_vec)} must match arr.ndim={ndim}")

    axes_list = _parse_axes(axes, ndim)

    # Dispatch on dtype — uint8 and float32 accepted natively (2× faster for f32)
    if arr.dtype == np.uint8 and arr.flags['C_CONTIGUOUS']:
        return compute_u8(arr, axes_list, stride_vec, n_moments)
    if arr.dtype == np.float32 and arr.flags['C_CONTIGUOUS']:
        return compute_f32(arr, axes_list, stride_vec, n_moments)
    # Everything else → float64
    arr64 = np.ascontiguousarray(arr, dtype=np.float64)
    return compute_f64(arr64, axes_list, stride_vec, n_moments)


__all__ = ["compute"]
