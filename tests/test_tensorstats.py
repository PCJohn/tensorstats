"""
tensorstats test suite.

Run from repo root:
    python -m pytest tests/ -v -s

All tests go through ts.StatsComputer — the sole public interface.

Output convention: moments-LAST.
  result["global"]  shape (n_moments,)
  result["0,1"]     shape (C, n_moments)
  result["grid_0"]    shape (*cell_shape, n_moments)
"""

import time
import numpy as np
import pytest
import tensorstats as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sc(arr, **kwargs):
    """Construct StatsComputer for arr.shape, compute once, copy grid levels."""
    result = ts.StatsComputer(shape=arr.shape, **kwargs).compute(arr)
    for k in list(result):
        if k.startswith("grid_"):
            result[k] = result[k].copy()
    return result


def numpy_moments(arr, axes=None):
    """Reference: two-pass central moments via numpy. Returns shape (*out, 4)."""
    a = arr.astype(np.float64)
    if axes is None:
        flat = a.ravel()
        mu = flat.mean()
        d = flat - mu
        d2 = d * d
        return np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
    ax = (axes,) if isinstance(axes, int) else tuple(axes)
    other = [dd for dd in range(a.ndim) if dd not in ax]
    flat = np.moveaxis(a, ax, range(len(ax))).reshape(
        -1, *[a.shape[dd] for dd in other]
    )
    mu = flat.mean(axis=0)
    d = flat - mu
    d2 = d * d
    return np.stack(
        [mu, d2.mean(axis=0), (d2 * d).mean(axis=0), (d2 * d2).mean(axis=0)], axis=-1
    )


def timeit_ms(fn, n=2000, warmup=50):
    """
    Return minimum per-call time in ms.
    Uses chunk sizes so each chunk runs ~20ms — OS scheduling noise is small.
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    per_call = (time.perf_counter() - t0) / 10 * 1000
    chunk = max(10, int(20.0 / per_call))
    n_chunks = max(5, n // chunk)
    best = float("inf")
    for _ in range(n_chunks):
        t0 = time.perf_counter()
        for _ in range(chunk):
            fn()
        best = min(best, (time.perf_counter() - t0) / chunk * 1000)
    return best


# ---------------------------------------------------------------------------
# Correctness — global and per-axis reductions
# ---------------------------------------------------------------------------


class TestCorrectness:

    def test_global_1d(self):
        arr = np.random.default_rng(0).standard_normal(1000)
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-10)

    def test_global_image_uint8(self):
        arr = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-6)

    def test_global_float32(self):
        arr = np.random.default_rng(2).random((64, 64, 3), dtype=np.float32)
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-5)

    def test_per_channel_shape_and_values(self):
        arr = np.random.default_rng(3).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = sc(arr, axes=(0, 1))
        assert out["0,1"].shape == (3, 4)
        np.testing.assert_allclose(
            out["0,1"], numpy_moments(arr, axes=(0, 1)), rtol=1e-8
        )

    def test_per_channel_indexing(self):
        """Moments-last: channel means = result["0,1"][:, 0]"""
        arr = np.random.default_rng(4).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = sc(arr, axes=(0, 1))
        for c in range(3):
            np.testing.assert_allclose(out["0,1"][c, 0], arr[:, :, c].mean(), rtol=1e-8)

    def test_axis_0_shape(self):
        arr = np.random.default_rng(5).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = sc(arr, axes=0)
        assert out["0"].shape == (64, 3, 4)

    def test_multi_axes_matches_individual(self):
        arr = np.random.default_rng(6).integers(0, 255, (64, 64, 3)).astype(np.float64)
        combined = sc(arr, axes=[None, (0, 1), 0, 1, 2])
        for axes, key in [
            (None, "global"),
            ((0, 1), "0,1"),
            (0, "0"),
            (1, "1"),
            (2, "2"),
        ]:
            solo = sc(arr, axes=axes)
            np.testing.assert_array_equal(combined[key], solo[key])

    def test_near_constant_stable(self):
        arr = (
            np.full(64 * 64 * 3, 128.0)
            + np.random.default_rng(7).standard_normal(64 * 64 * 3) * 0.01
        )
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-6)

    def test_large_values(self):
        arr = np.random.default_rng(8).uniform(0, 1e6, (64, 64, 3))
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-8)

    def test_n_moments_1_to_4(self):
        arr = np.random.default_rng(9).integers(0, 255, (32, 32, 3)).astype(np.float64)
        ref = numpy_moments(arr)
        for k in range(1, 5):
            out = sc(arr, axes=None, n_moments=k)
            assert out["global"].shape == (k,)
            np.testing.assert_allclose(out["global"], ref[:k], rtol=1e-8)

    def test_output_shapes(self):
        arr = np.zeros((288, 512, 3), dtype=np.float64)
        out = sc(arr, axes=[None, (0, 1), 0, 1, 2])
        assert out["global"].shape == (4,)
        assert out["0,1"].shape == (3, 4)
        assert out["0"].shape == (512, 3, 4)
        assert out["1"].shape == (288, 3, 4)
        assert out["2"].shape == (288, 512, 4)

    def test_grayscale_2d(self):
        arr = np.random.default_rng(10).integers(0, 255, (64, 64)).astype(np.float64)
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-8)

    def test_native_uint8_matches_float64(self):
        arr = np.random.default_rng(11).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        u8 = sc(arr, axes=[None, (0, 1)])
        f64 = sc(arr.astype(np.float64), axes=[None, (0, 1)])
        np.testing.assert_allclose(u8["global"], f64["global"], rtol=1e-6)
        np.testing.assert_allclose(u8["0,1"], f64["0,1"], rtol=1e-6)

    def test_native_float32_matches_float64(self):
        arr = np.random.default_rng(12).random((64, 64, 3), dtype=np.float32)
        f32 = sc(arr, axes=None)
        f64 = sc(arr.astype(np.float64), axes=None)
        np.testing.assert_allclose(f32["global"], f64["global"], rtol=1e-5)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class TestGrid:

    def _numpy_grid_ref(self, arr, grid):
        """Reference: compute moments per grid cell using numpy."""
        ndim = arr.ndim
        n_cells = [2**g for g in grid]
        out_shape = tuple(n_cells) + (4,)
        result = np.zeros(out_shape)

        def recurse(d, cell_coords, slices):
            if d == ndim:
                cell_data = arr[tuple(slices)].ravel().astype(np.float64)
                mu = cell_data.mean()
                dv = cell_data - mu
                d2 = dv * dv
                result[tuple(cell_coords)] = [
                    mu,
                    d2.mean(),
                    (d2 * dv).mean(),
                    (d2 * d2).mean(),
                ]
                return
            n = arr.shape[d]
            nc = n_cells[d]
            for ci in range(nc):
                lo = ci * n // nc
                hi = (ci + 1) * n // nc
                recurse(d + 1, cell_coords + [ci], slices + [slice(lo, hi)])

        recurse(0, [], [])
        return result

    def test_grid_shape_2x2(self):
        arr = np.random.default_rng(20).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(1, 1, 0))
        assert out["grid_0"].shape == (2, 2, 1, 4)

    def test_grid_shape_4x4(self):
        arr = np.random.default_rng(21).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(2, 2, 0))
        assert out["grid_0"].shape == (4, 4, 1, 4)

    def test_grid_shape_8x8(self):
        arr = np.random.default_rng(22).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(3, 3, 0))
        assert out["grid_0"].shape == (8, 8, 1, 4)

    def test_grid_correctness_2x2(self):
        arr = np.random.default_rng(23).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(1, 1, 0))
        ref = self._numpy_grid_ref(arr, (1, 1, 0))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6)

    def test_grid_correctness_4x4(self):
        arr = np.random.default_rng(24).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(2, 2, 0))
        ref = self._numpy_grid_ref(arr, (2, 2, 0))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6)

    def test_grid_correctness_8x8(self):
        arr = np.random.default_rng(25).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(3, 3, 0))
        ref = self._numpy_grid_ref(arr, (3, 3, 0))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6)

    def test_grid_correctness_64x64(self):
        arr = np.random.default_rng(26).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(4, 4, 0))
        ref = self._numpy_grid_ref(arr, (4, 4, 0))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6)

    def test_grid_float64(self):
        arr = np.random.default_rng(27).random((64, 64, 3))
        out = sc(arr, axes=None, grid=(2, 2, 0))
        ref = self._numpy_grid_ref(arr, (2, 2, 0))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-8)

    def test_grid_moments_last_indexing(self):
        """grid[r, c, ch, 0] = mean of that cell. grid=(2,2,0) → 4x4x1 cells."""
        arr = np.random.default_rng(28).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(2, 2, 0))
        assert out["grid_0"].shape == (4, 4, 1, 4)
        # Cell [0,0,0]: top-left 16x16x3 patch (64/4=16 pixels per spatial cell)
        expected_mean = arr[:16, :16, :].mean(dtype=np.float64)
        np.testing.assert_allclose(out["grid_0"][0, 0, 0, 0], expected_mean, rtol=1e-6)

    def test_grid_alongside_axes(self):
        """Grid and axes can be computed in the same call."""
        arr = np.random.default_rng(29).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=[None, (0, 1)], grid=(1, 1, 0))  # 2x2x1 cells
        assert "global" in out and "0,1" in out and "grid_0" in out
        assert out["grid_0"].shape == (2, 2, 1, 4)

    def test_grid_1d_tensor(self):
        arr = np.random.default_rng(30).integers(0, 255, (256,), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(3,))
        ref = self._numpy_grid_ref(arr, (3,))
        assert out["grid_0"].shape == (8, 4)
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6)

    def test_grid_no_subdivision(self):
        """grid=(0,0,0) → 1 cell covering the whole array."""
        arr = np.random.default_rng(31).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(0, 0, 0))
        ref = numpy_moments(arr)
        assert out["grid_0"].shape == (1, 1, 1, 4)
        np.testing.assert_allclose(out["grid_0"][0, 0, 0], ref, rtol=1e-6)

    def test_grid_stateful_retained_buffer(self):
        """StatsComputer returns independent copies each call — consecutive calls are safe."""
        arr1 = np.random.default_rng(32).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        arr2 = np.random.default_rng(33).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        computer = ts.StatsComputer(shape=(64, 64, 3), axes=None, grid=(2, 2, 0))
        ref1 = self._numpy_grid_ref(arr1, (2, 2, 0))
        ref2 = self._numpy_grid_ref(arr2, (2, 2, 0))
        g1 = computer.compute(arr1)["grid_0"]
        g2 = computer.compute(arr2)["grid_0"]
        # Both results are independent copies — g1 is unaffected by second call
        np.testing.assert_allclose(g1, ref1, rtol=1e-6)
        np.testing.assert_allclose(g2, ref2, rtol=1e-6)

    @pytest.mark.parametrize("exp", [1, 2, 3, 4, 5, 6])
    def test_grid_latency(self, exp):
        arr = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        n_cells = (2**exp) ** 2
        computer = ts.StatsComputer(shape=arr.shape, axes=None, grid=(exp, exp, 0))

        def numpy_grid():
            n = 2**exp
            for r in range(n):
                for c in range(n):
                    h, w = 64 // n, 64 // n
                    patch = arr[r * h : (r + 1) * h, c * w : (c + 1) * w, :]
                    _ = numpy_moments(patch)

        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(numpy_grid)
        print(
            f"\n  grid={2**exp}x{2**exp} ({n_cells} cells)  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f}"


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------


class TestStride:

    def test_stride_1_matches_no_stride(self):
        arr = np.random.default_rng(40).random((64, 64, 3))
        out1 = sc(arr, axes=None)
        out2 = sc(arr, axes=None, stride=1)
        np.testing.assert_allclose(out1["global"], out2["global"], rtol=1e-10)

    def test_stride_tuple_global(self):
        arr = np.random.default_rng(41).random((8, 8, 3))
        out = sc(arr, axes=None, stride=(2, 2, 1))
        ref_arr = arr[::2, ::2, :].ravel().astype(np.float64)
        mu = ref_arr.mean()
        d = ref_arr - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)

    def test_stride_scalar_global(self):
        arr = np.random.default_rng(42).standard_normal(256)
        out = sc(arr, axes=None, stride=2)
        ref_arr = arr[::2].astype(np.float64)
        mu = ref_arr.mean()
        d = ref_arr - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-10)

    def test_stride_per_channel(self):
        arr = np.random.default_rng(43).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = sc(arr, axes=(0, 1), stride=(2, 2, 1))
        for c in range(3):
            ref = numpy_moments(arr[::2, ::2, c])
            np.testing.assert_allclose(out["0,1"][c], ref, rtol=1e-8)

    @pytest.mark.parametrize("stride", [1, 2, 4, 8])
    def test_stride_accuracy_vs_numpy(self, stride):
        """
        Stride subsamples the input — this is intentional and introduces
        approximation error that grows with stride. The test verifies the
        computation is internally consistent (matches numpy on the same
        subsampled pixels), not that it matches the full-resolution result.
        """
        arr = np.random.default_rng(44).random((64, 64, 3))
        out = sc(arr, axes=None, stride=stride)
        # Scalar stride steps through the flat array — not per-axis subsampling
        sub = arr.ravel()[::stride].astype(np.float64)
        mu = sub.mean()
        d = sub - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        errs = np.abs(out["global"] - ref) / (np.abs(ref) + 1e-10)
        names = ["mean", "var", "m3", "m4"]
        print(
            f"\n  stride={stride}:  "
            + "  ".join(f"{n}_err={e:.2e}" for n, e in zip(names, errs))
        )
        # Exact agreement with the subsampled reference (not an approximation)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# Fused last-axis (per-channel) path under stride.
#
# Reducing all-but-last with any stride routes through last_axis_fused /
# last_axis_u8_fused: a single strided walk over the leading axes that collects
# every channel at once. uint8 stays on the histogram path even with multiple
# strided leading axes (previously this fell back to a gather). These tests pin
# correctness against numpy on the same subsampled pixels.
# ---------------------------------------------------------------------------


class TestFusedStridePerChannel:

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_multi_axis_stride(self, dtype):
        rng = np.random.default_rng(101)
        arr = rng.integers(0, 255, (96, 80, 3)).astype(dtype)
        out = sc(arr, axes=(0, 1), stride=(2, 2, 1))
        for c in range(3):
            ref = numpy_moments(arr[::2, ::2, c])
            np.testing.assert_allclose(out["0,1"][c], ref, rtol=1e-7, atol=1e-9)

    def test_uint8_matches_float64(self):
        """Histogram path and direct two-pass must agree exactly on the same
        integer pixels."""
        rng = np.random.default_rng(102)
        base = rng.integers(0, 255, (128, 128, 3))
        u8 = sc(base.astype(np.uint8), axes=(0, 1), stride=(2, 2, 1))["0,1"]
        f64 = sc(base.astype(np.float64), axes=(0, 1), stride=(2, 2, 1))["0,1"]
        np.testing.assert_allclose(u8, f64, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float64])
    def test_four_d_three_leading_axes(self, dtype):
        """Odometer walk over three strided leading axes (4D input)."""
        rng = np.random.default_rng(103)
        arr = rng.integers(0, 255, (24, 20, 16, 3)).astype(dtype)
        out = sc(arr, axes=(0, 1, 2), stride=(2, 2, 2, 1))
        for c in range(3):
            ref = numpy_moments(arr[::2, ::2, ::2, c])
            np.testing.assert_allclose(out["0,1,2"][c], ref, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float64])
    def test_single_leading_axis_stride(self, dtype):
        """axes=(0,) on a 2D array — one strided leading axis."""
        rng = np.random.default_rng(104)
        arr = rng.integers(0, 255, (200, 5)).astype(dtype)
        out = sc(arr, axes=(0,), stride=(3, 1))
        for c in range(5):
            ref = numpy_moments(arr[::3, c])
            np.testing.assert_allclose(out["0"][c], ref, rtol=1e-7, atol=1e-9)

    def test_uint8_faster_than_float64(self):
        """With the histogram path restored, strided per-channel uint8 must beat
        the float two-pass."""
        arr = np.random.default_rng(105).integers(0, 255, (256, 256, 3))
        cu = ts.StatsComputer(shape=arr.shape, axes=(0, 1), stride=(2, 2, 1))
        cf = ts.StatsComputer(shape=arr.shape, axes=(0, 1), stride=(2, 2, 1))
        au, af = arr.astype(np.uint8), arr.astype(np.float64)
        u8_ms = timeit_ms(lambda: cu.compute(au))
        f64_ms = timeit_ms(lambda: cf.compute(af))
        print(f"\n  strided perchan 256x256x3: u8={u8_ms:.4f}ms  f64={f64_ms:.4f}ms")
        assert u8_ms < f64_ms


# ---------------------------------------------------------------------------
# uint8 histogram path
# ---------------------------------------------------------------------------


class TestUint8:

    def _ref_moments(self, arr, axes=None):
        return numpy_moments(arr, axes)

    def test_all_zeros(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None)
        assert out["global"][0] == 0.0
        assert out["global"][1] == 0.0

    def test_all_255(self):
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        out = sc(arr, axes=None)
        assert out["global"][0] == 255.0
        assert out["global"][1] == 0.0

    def test_flat_value_zero_variance(self):
        for v in [0, 1, 127, 128, 254, 255]:
            arr = np.full((32, 32, 3), v, dtype=np.uint8)
            out = sc(arr, axes=None)
            assert out["global"][1] == 0.0, f"value={v}"

    CASES = {
        "random": lambda rng: rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
        "all_zeros": lambda rng: np.zeros((64, 64, 3), dtype=np.uint8),
        "all_255": lambda rng: np.full((64, 64, 3), 255, dtype=np.uint8),
        "flat_128": lambda rng: np.full((64, 64, 3), 128, dtype=np.uint8),
        "gaussian_blob": lambda rng: np.clip(
            rng.normal(128, 30, (64, 64, 3)), 0, 255
        ).astype(np.uint8),
        "horizontal_gradient": lambda rng: np.tile(
            np.arange(64, dtype=np.uint8).reshape(1, 64, 1), (64, 1, 3)
        ),
        "two_tone": lambda rng: (rng.integers(0, 2, (64, 64, 3)) * 255).astype(
            np.uint8
        ),
        "near_constant": lambda rng: np.clip(
            rng.integers(127, 130, (64, 64, 3)), 0, 255
        ).astype(np.uint8),
        "checkerboard": lambda rng: (
            ((np.arange(64).reshape(-1, 1) + np.arange(64).reshape(1, -1)) % 2)[
                :, :, None
            ]
            * np.ones((1, 1, 3))
        ).astype(np.uint8)
        * 255,
    }

    @pytest.mark.parametrize("name", list(CASES.keys()))
    def test_matches_numpy_all_cases(self, name):
        arr = self.CASES[name](np.random.default_rng(50))
        out = sc(arr, axes=None)
        ref = self._ref_moments(arr)
        np.testing.assert_allclose(
            out["global"], ref, rtol=1e-5, err_msg=f"case={name}"
        )

    @pytest.mark.parametrize("name", list(CASES.keys()))
    def test_per_channel_matches_numpy_all_cases(self, name):
        arr = self.CASES[name](np.random.default_rng(51))
        out = sc(arr, axes=(0, 1))
        ref = self._ref_moments(arr, axes=(0, 1))
        np.testing.assert_allclose(out["0,1"], ref, rtol=1e-5, err_msg=f"case={name}")

    @pytest.mark.parametrize("stride", [1, 2, 4, 8])
    def test_stride_accuracy_uint8(self, stride):
        arr = np.random.default_rng(52).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, stride=stride)
        sub = arr.ravel()[::stride].astype(np.float64)
        mu = sub.mean()
        d = sub - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        errs = np.abs(out["global"] - ref) / (np.abs(ref) + 1e-10)
        names = ["mean", "var", "m3", "m4"]
        print(
            f"\n  uint8 stride={stride}:  "
            + "  ".join(f"{n}_err={e:.2e}" for n, e in zip(names, errs))
        )
        np.testing.assert_allclose(out["global"], ref, rtol=1e-4)

    def test_histogram_faster_than_numpy_global(self):
        arr = np.random.default_rng(53).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        computer = ts.StatsComputer(shape=arr.shape, axes=None)
        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(lambda: numpy_moments(arr))
        print(
            f"\n  u8 hist 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        assert ts_ms < np_ms

    def test_histogram_faster_than_numpy_per_channel(self):
        arr = np.random.default_rng(54).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        computer = ts.StatsComputer(shape=arr.shape, axes=(0, 1))
        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(lambda: numpy_moments(arr, axes=(0, 1)))
        print(
            f"\n  u8 hist perchan 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        assert ts_ms < np_ms


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------

SHAPES = [(16, 16, 3), (32, 32, 3), (64, 64, 3), (128, 128, 3), (256, 256, 3)]


class TestLatency:

    def _numpy_global(self, arr):
        return numpy_moments(arr)

    def _numpy_multi(self, arr):
        return numpy_moments(arr, axes=(0, 1))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_float64_faster_than_numpy(self, shape):
        """
        ts wins at small sizes (≤32×32) due to call-overhead advantage.
        At larger sizes numpy's BLAS/SIMD can win depending on platform
        (especially on Mac with Accelerate, and at L2/L3 cache boundaries).
        We assert only where the win is reliable across all platforms.
        """
        arr = (
            np.random.default_rng(0)
            .integers(0, 255, shape, dtype=np.uint8)
            .astype(np.float64)
        )
        computer = ts.StatsComputer(shape=shape, axes=None)
        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(lambda: self._numpy_global(arr))
        print(
            f"\n  shape={shape} f64  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        h, w, _ = shape
        if h * w <= 32 * 32:
            assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape}"

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_uint8_faster_than_numpy(self, shape):
        arr = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
        computer = ts.StatsComputer(shape=shape, axes=None)
        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(lambda: self._numpy_global(arr))
        print(
            f"\n  shape={shape} u8   ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        h, w, _ = shape
        if h * w <= 32 * 32:
            assert ts_ms < np_ms

    @pytest.mark.parametrize("shape", SHAPES)
    def test_multichannel_faster_than_numpy(self, shape):
        arr = np.random.default_rng(2).integers(0, 255, shape, dtype=np.uint8)
        computer = ts.StatsComputer(shape=shape, axes=[None, (0, 1)])
        ts_ms = timeit_ms(lambda: computer.compute(arr))
        np_ms = timeit_ms(lambda: self._numpy_multi(arr))
        print(
            f"\n  shape={shape} multi  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x"
        )
        h, w, _ = shape
        if h * w <= 64 * 64:
            assert ts_ms < np_ms

    @pytest.mark.parametrize("shape", SHAPES)
    def test_stride2_faster_than_no_stride(self, shape):
        arr = np.random.default_rng(3).integers(0, 255, shape, dtype=np.uint8)
        c1 = ts.StatsComputer(shape=shape, axes=None)
        c2 = ts.StatsComputer(shape=shape, axes=None, stride=2)
        ts_ns = timeit_ms(lambda: c1.compute(arr))
        ts_s2 = timeit_ms(lambda: c2.compute(arr))
        print(
            f"\n  shape={shape} stride  nostride={ts_ns:.4f}ms  stride2={ts_s2:.4f}ms  ratio={ts_ns/ts_s2:.2f}x"
        )
        h, w, _ = shape
        if h * w >= 64 * 64:
            assert ts_s2 < ts_ns

    @pytest.mark.parametrize("shape", SHAPES)
    def test_statscomputer_faster_than_fresh(self, shape):
        """StatsComputer.compute() must be faster than constructing fresh each call."""
        arr = np.random.default_rng(4).integers(0, 255, shape, dtype=np.uint8)
        computer = ts.StatsComputer(shape=shape, axes=[None, (0, 1)], grid=(3, 3, 0))
        stateful_ms = timeit_ms(lambda: computer.compute(arr))
        fresh_ms = timeit_ms(
            lambda: ts.StatsComputer(
                shape=shape, axes=[None, (0, 1)], grid=(3, 3, 0)
            ).compute(arr)
        )
        print(
            f"\n  shape={shape} stateful={stateful_ms:.4f}ms  fresh={fresh_ms:.4f}ms  ratio={fresh_ms/stateful_ms:.2f}x"
        )
        assert stateful_ms < fresh_ms, "stateful should be faster (retained buffers)"


# ---------------------------------------------------------------------------
# Higher-dimensional tensor tests (2D through 6D)
# ---------------------------------------------------------------------------


class TestHigherDim:

    def _moments_ref(self, arr, axes=None):
        a = arr.astype(np.float64)
        if axes is None:
            flat = a.ravel()
            mu = flat.mean()
            d = flat - mu
            d2 = d * d
            return np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        ax = (axes,) if isinstance(axes, int) else tuple(axes)
        other = [dd for dd in range(a.ndim) if dd not in ax]
        flat = np.moveaxis(a, ax, range(len(ax))).reshape(
            -1, *[a.shape[dd] for dd in other]
        )
        mu = flat.mean(axis=0)
        d = flat - mu
        d2 = d * d
        return np.stack(
            [mu, d2.mean(axis=0), (d2 * d).mean(axis=0), (d2 * d2).mean(axis=0)],
            axis=-1,
        )

    def test_2d_global(self):
        arr = np.random.default_rng(100).integers(0, 255, (64, 64), dtype=np.uint8)
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-6)

    def test_2d_axis0(self):
        arr = np.random.default_rng(101).random((64, 64))
        out = sc(arr, axes=0)
        ref = self._moments_ref(arr, axes=0)
        assert out["0"].shape == (64, 4)
        np.testing.assert_allclose(out["0"], ref, rtol=1e-8)

    def test_2d_grid(self):
        arr = np.random.default_rng(102).integers(0, 255, (64, 64), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(2, 2))
        assert out["grid_0"].shape == (4, 4, 4)
        ref_cell = self._moments_ref(arr[:16, :16])
        np.testing.assert_allclose(out["grid_0"][0, 0], ref_cell, rtol=1e-6)

    def test_4d_global(self):
        arr = np.random.default_rng(110).random((4, 8, 8, 3))
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_4d_reduce_last_two(self):
        arr = np.random.default_rng(111).random((4, 8, 8, 3))
        out = sc(arr, axes=(1, 2))
        ref = self._moments_ref(arr, axes=(1, 2))
        assert out["1,2"].shape == (4, 3, 4)
        np.testing.assert_allclose(out["1,2"], ref, rtol=1e-8)

    def test_4d_reduce_first(self):
        arr = np.random.default_rng(112).random((4, 8, 8, 3))
        out = sc(arr, axes=0)
        ref = self._moments_ref(arr, axes=0)
        assert out["0"].shape == (8, 8, 3, 4)
        np.testing.assert_allclose(out["0"], ref, rtol=1e-8)

    def test_4d_grid(self):
        arr = np.random.default_rng(113).integers(
            0, 255, (4, 16, 16, 3), dtype=np.uint8
        )
        out = sc(arr, axes=None, grid=(1, 2, 2, 0))
        assert out["grid_0"].shape == (2, 4, 4, 1, 4)

    def test_4d_multiple_axes(self):
        arr = np.random.default_rng(114).random((3, 4, 5, 6))
        out = sc(arr, axes=[None, (0, 1), (2, 3)])
        assert out["global"].shape == (4,)
        assert out["0,1"].shape == (5, 6, 4)
        assert out["2,3"].shape == (3, 4, 4)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)
        np.testing.assert_allclose(
            out["0,1"], self._moments_ref(arr, axes=(0, 1)), rtol=1e-8
        )

    def test_5d_global(self):
        arr = np.random.default_rng(120).random((2, 3, 4, 5, 6))
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_5d_reduce_middle(self):
        arr = np.random.default_rng(121).random((2, 4, 4, 4, 3))
        out = sc(arr, axes=(1, 2, 3))
        ref = self._moments_ref(arr, axes=(1, 2, 3))
        assert out["1,2,3"].shape == (2, 3, 4)
        np.testing.assert_allclose(out["1,2,3"], ref, rtol=1e-8)

    def test_5d_uint8(self):
        arr = np.random.default_rng(122).integers(
            0, 255, (2, 4, 4, 4, 3), dtype=np.uint8
        )
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-6)

    def test_6d_global(self):
        arr = np.random.default_rng(130).random((2, 3, 4, 4, 4, 2))
        out = sc(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_6d_reduce_inner(self):
        arr = np.random.default_rng(131).random((2, 2, 4, 4, 4, 3))
        out = sc(arr, axes=(2, 3, 4))
        ref = self._moments_ref(arr, axes=(2, 3, 4))
        assert out["2,3,4"].shape == (2, 2, 3, 4)
        np.testing.assert_allclose(out["2,3,4"], ref, rtol=1e-8)

    def test_6d_grid(self):
        arr = np.random.default_rng(132).integers(
            0, 255, (2, 2, 4, 4, 4, 3), dtype=np.uint8
        )
        out = sc(arr, axes=None, grid=(1, 1, 1, 1, 1, 0))
        assert out["grid_0"].shape == (2, 2, 2, 2, 2, 1, 4)

    def test_stride_4d(self):
        arr = np.random.default_rng(140).random((8, 8, 8, 8))
        out = sc(arr, axes=None, stride=2)
        flat = arr.ravel()[::2].astype(np.float64)
        mu = flat.mean()
        d = flat - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)

    def test_stride_tuple_4d(self):
        arr = np.random.default_rng(141).random((8, 8, 8, 3))
        out = sc(arr, axes=None, stride=(2, 2, 2, 1))
        ref_arr = arr[::2, ::2, ::2, :].ravel().astype(np.float64)
        mu = ref_arr.mean()
        d = ref_arr - mu
        d2 = d * d
        ref = np.array([mu, d2.mean(), (d2 * d).mean(), (d2 * d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)


# ---------------------------------------------------------------------------
# Multi-grid pyramid + stride-on-grid (added with the multi-grid extension)
# ---------------------------------------------------------------------------


def grid_ref(arr, grid, stride=None):
    """Reference per-cell central moments matching the C++ grid semantics:
    full-resolution cell boundaries (coord * n_cells // shape) with an optional
    uniform stride mask (a pixel is visited iff every axis coord is a multiple
    of that axis's stride). Returns shape (*n_cells, 4)."""
    a = arr.astype(np.float64)
    ndim = a.ndim
    if stride is None:
        stride = [1] * ndim
    n_cells = [2**g for g in grid]
    cstride = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        cstride[d] = cstride[d + 1] * n_cells[d + 1]
    cid = np.zeros(a.shape, dtype=np.int64)
    mask = np.ones(a.shape, dtype=bool)
    for d in range(ndim):
        shp = [1] * ndim
        shp[d] = a.shape[d]
        lut = (np.arange(a.shape[d]) * n_cells[d] // a.shape[d]).reshape(shp)
        cid += lut * cstride[d]
        mv = np.zeros(a.shape[d], dtype=bool)
        mv[:: stride[d]] = True
        mask &= mv.reshape(shp)
    out = np.zeros(int(np.prod(n_cells)) * 4).reshape(-1, 4)
    cidv = cid[mask]
    av = a[mask]
    order = np.argsort(cidv, kind="stable")
    cidv, av = cidv[order], av[order]
    uniq, start = np.unique(cidv, return_index=True)
    for c, vals in zip(uniq, np.split(av, start[1:])):
        mu = vals.mean()
        dv = vals - mu
        d2 = dv * dv
        out[c] = [mu, d2.mean(), (d2 * dv).mean(), (d2 * d2).mean()]
    return out.reshape(tuple(n_cells) + (4,))


class TestGridPyramid:
    """Multiple grid resolutions computed in a single pass."""

    PYRAMID = [(1, 1, 0), (2, 2, 0), (3, 3, 0)]  # 2x2, 4x4, 8x8 spatial

    def test_single_tuple_is_grid_0(self):
        """Back-compat: a single tuple still works, now keyed grid_0."""
        arr = np.random.default_rng(300).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=(2, 2, 0))
        assert "grid_0" in out and "grid_1" not in out
        np.testing.assert_allclose(out["grid_0"], grid_ref(arr, (2, 2, 0)), rtol=1e-6)

    def test_pyramid_keys_and_order(self):
        arr = np.random.default_rng(301).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=self.PYRAMID)
        assert [k for k in out if k.startswith("grid_")] == [
            "grid_0",
            "grid_1",
            "grid_2",
        ]
        for i, g in enumerate(self.PYRAMID):
            assert out[f"grid_{i}"].shape == tuple(2**k for k in g) + (4,)

    def test_pyramid_uint8_matches_per_level(self):
        arr = np.random.default_rng(302).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=None, grid=self.PYRAMID)
        for i, g in enumerate(self.PYRAMID):
            np.testing.assert_allclose(out[f"grid_{i}"], grid_ref(arr, g), rtol=1e-6)

    def test_pyramid_float64_matches_per_level(self):
        arr = np.random.default_rng(303).random((64, 64, 3))
        out = sc(arr, axes=None, grid=self.PYRAMID)
        for i, g in enumerate(self.PYRAMID):
            np.testing.assert_allclose(out[f"grid_{i}"], grid_ref(arr, g), rtol=1e-8)

    def test_pyramid_equals_separate_computers(self):
        """Fused multi-grid must give identical results to one computer per grid."""
        arr = np.random.default_rng(304).integers(0, 255, (96, 96, 3), dtype=np.uint8)
        fused = sc(arr, axes=None, grid=self.PYRAMID)
        for i, g in enumerate(self.PYRAMID):
            solo = sc(arr, axes=None, grid=g)["grid_0"]
            np.testing.assert_array_equal(fused[f"grid_{i}"], solo)

    def test_pyramid_alongside_axes(self):
        arr = np.random.default_rng(305).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = sc(arr, axes=[None, (0, 1)], grid=self.PYRAMID)
        assert {"global", "0,1", "grid_0", "grid_1", "grid_2"} <= set(out)

    def test_pyramid_level_independence_across_calls(self):
        """Retained per-level buffers must not alias across consecutive calls."""
        a1 = np.random.default_rng(306).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        a2 = np.random.default_rng(307).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        comp = ts.StatsComputer(shape=a1.shape, axes=None, grid=self.PYRAMID)
        r1 = {k: v.copy() for k, v in comp.compute(a1).items()}
        r2 = {k: v.copy() for k, v in comp.compute(a2).items()}
        for i, g in enumerate(self.PYRAMID):
            np.testing.assert_allclose(r1[f"grid_{i}"], grid_ref(a1, g), rtol=1e-6)
            np.testing.assert_allclose(r2[f"grid_{i}"], grid_ref(a2, g), rtol=1e-6)


class TestGridStride:
    """Stride applied to the grid path — exact moments over the subsampled
    pixels of each cell. Mirrors grid_ref's full-resolution-boundary semantics."""

    SIZES = [(64, 64, 3), (256, 256, 3), (1024, 1024, 3)]
    STRIDES = [1, 2, 4, 8]

    def _check(self, arr, grid, stride):
        out = sc(arr, axes=None, grid=grid, stride=stride)
        ref = grid_ref(arr, grid, stride=list(stride))
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("stride", STRIDES)
    def test_zeros(self, stride):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        self._check(arr, (2, 2, 0), (stride, stride, 1))

    @pytest.mark.parametrize("stride", STRIDES)
    def test_full_255(self, stride):
        arr = np.full((256, 256, 3), 255, dtype=np.uint8)
        self._check(arr, (2, 2, 0), (stride, stride, 1))

    @pytest.mark.parametrize("stride", STRIDES)
    def test_random_uint8(self, stride):
        arr = np.random.default_rng(400).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        self._check(arr, (3, 3, 0), (stride, stride, 1))

    @pytest.mark.parametrize("stride", STRIDES)
    def test_random_float64(self, stride):
        arr = np.random.default_rng(401).random((256, 256, 3))
        self._check(arr, (3, 3, 0), (stride, stride, 1))

    @pytest.mark.parametrize("shape", SIZES)
    def test_gaussian_blob_sizes(self, shape):
        """Gaussian blob across sizes (incl. large), strided grid, uint8 + float."""
        rng = np.random.default_rng(402)
        blob = np.clip(rng.normal(128, 30, shape), 0, 255)
        for arr in (blob.astype(np.uint8), blob.astype(np.float64)):
            self._check(arr, (2, 2, 0), (4, 4, 1))

    def test_gaussian_blob_known_per_channel(self):
        """Distinct per-channel distributions — strided per-channel stats exact
        vs numpy on the same subsampled pixels."""
        rng = np.random.default_rng(403)
        chans = [
            rng.normal(m, s, (256, 256)) for m, s in [(50, 5), (128, 20), (200, 3)]
        ]
        arr = np.stack(chans, axis=-1)
        out = sc(arr, axes=(0, 1), stride=(2, 2, 1))
        for c in range(3):
            np.testing.assert_allclose(
                out["0,1"][c], numpy_moments(arr[::2, ::2, c]), rtol=1e-8
            )

    def test_gaussian_blob_known_per_cell(self):
        """Each spatial cell drawn from its own distribution — verifies cells do
        not bleed and per-cell moments are exact, with and without stride."""
        rng = np.random.default_rng(404)
        H = W = 128
        nc = 4  # 4x4 spatial cells -> grid (2,2,0)
        arr = np.zeros((H, W, 1))
        for r in range(nc):
            for c in range(nc):
                k = r * nc + c
                arr[
                    r * H // nc : (r + 1) * H // nc, c * W // nc : (c + 1) * W // nc
                ] = rng.normal(10 * k + 20, 1 + 0.3 * k, (H // nc, W // nc, 1))
        for stride in [(1, 1, 1), (2, 2, 1)]:
            out = sc(arr, axes=None, grid=(2, 2, 0), stride=stride)
            np.testing.assert_allclose(
                out["grid_0"], grid_ref(arr, (2, 2, 0), stride=list(stride)), rtol=1e-9
            )
        # cell means track their generating loc
        out = sc(arr, axes=None, grid=(2, 2, 0))
        for r in range(nc):
            for c in range(nc):
                assert abs(out["grid_0"][r, c, 0, 0] - (10 * (r * nc + c) + 20)) < 1.0

    def test_stride_empty_cell_is_zero(self):
        """A grid finer than the strided sampling leaves empty cells -> zeros,
        not stale retained-buffer data."""
        arr = np.random.default_rng(405).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        out = sc(
            arr, axes=None, grid=(3, 3, 0), stride=(8, 8, 1)
        )  # 8x8 cells, 2x2 samples
        ref = grid_ref(arr, (3, 3, 0), stride=[8, 8, 1])
        np.testing.assert_allclose(out["grid_0"], ref, rtol=1e-6, atol=1e-12)


class TestGridPyramidLatency:
    """Latency sweep across sizes/strides; stride is the lever to the budget."""

    @pytest.mark.parametrize(
        "shape", [(64, 64, 3), (256, 256, 3), (1024, 1024, 3), (2048, 2048, 3)]
    )
    def test_pyramid_latency_sweep(self, shape):
        arr = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
        pyr = [(2, 2, 0), (3, 3, 0), (4, 4, 0)]

        def quick_ms(fn):
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            fn()
            est = max(time.perf_counter() - t0, 1e-5)
            iters = int(min(300, max(5, 0.05 / est)))  # ~50ms total wall per config
            best = float("inf")
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                best = min(best, (time.perf_counter() - t0) * 1000)
            return best

        print(f"\n  {shape} uint8 3-level pyramid:")
        for st in [1, 2, 4, 8]:
            comp = ts.StatsComputer(
                shape=shape, axes=None, grid=pyr, stride=(st, st, 1)
            )
            print(f"    stride={st}: {quick_ms(lambda: comp.compute(arr)):.4f}ms")

    def test_fused_not_slower_than_separate(self):
        arr = np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)
        pyr = [(2, 2, 0), (3, 3, 0), (4, 4, 0)]
        fused = ts.StatsComputer(shape=arr.shape, axes=None, grid=pyr)
        seps = [ts.StatsComputer(shape=arr.shape, axes=None, grid=g) for g in pyr]
        f_ms = timeit_ms(lambda: fused.compute(arr))
        s_ms = timeit_ms(lambda: [c.compute(arr) for c in seps])
        print(
            f"\n  128x128x3 pyramid fused={f_ms:.4f}ms separate={s_ms:.4f}ms ratio={s_ms/f_ms:.2f}x"
        )
        assert f_ms < s_ms

    def test_stride_reduces_latency(self):
        arr = np.random.default_rng(2).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        pyr = [(2, 2, 0), (3, 3, 0), (4, 4, 0)]
        c1 = ts.StatsComputer(shape=arr.shape, axes=None, grid=pyr, stride=1)
        c4 = ts.StatsComputer(shape=arr.shape, axes=None, grid=pyr, stride=(4, 4, 1))
        assert timeit_ms(lambda: c4.compute(arr)) < timeit_ms(lambda: c1.compute(arr))
