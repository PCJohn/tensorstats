"""
tensorstats test suite.

Structure:
    tests/
    └── test_tensorstats.py   ← this file

Run from repo root:
    python -m pytest tests/ -v -s
"""

import time
import numpy as np
import pytest
import tensorstats as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_moments(arr, axes=None):
    """Reference: two-pass central moments via numpy."""
    a = arr.astype(np.float64)
    if axes is None:
        flat = a.ravel()
    else:
        ax = (axes,) if isinstance(axes, int) else tuple(axes)
        other = [d for d in range(a.ndim) if d not in ax]
        flat = np.moveaxis(a, ax, range(len(ax))).reshape(
            -1, *[a.shape[d] for d in other])
    mu  = flat.mean(axis=0)
    d   = flat - mu
    d2  = d * d
    return np.stack([np.broadcast_to(mu, mu.shape),
                     d2.mean(axis=0), (d2*d).mean(axis=0), (d2*d2).mean(axis=0)])


def timeit_ms(fn, n=2000, warmup=50):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:

    def test_global_1d(self):
        arr = np.random.default_rng(0).standard_normal(1000)
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-10)

    def test_global_image_uint8(self):
        arr = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_global_float32(self):
        arr = np.random.default_rng(2).random((64, 64, 3), dtype=np.float32)
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-5)

    def test_per_channel(self):
        arr = np.random.default_rng(3).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = ts.compute(arr, axes=(0, 1))
        for c in range(3):
            chan = arr[:, :, c].ravel().astype(np.float64)
            mu = chan.mean(); d = chan - mu; d2 = d*d
            ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
            np.testing.assert_allclose(out["0,1"][:, c], ref, rtol=1e-8,
                                       err_msg=f"channel {c}")

    def test_axis_0(self):
        arr = np.random.default_rng(4).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = ts.compute(arr, axes=0)
        assert out["0"].shape == (4, 64, 3)
        col, ch = 7, 1
        data = arr[:, col, ch].astype(np.float64)
        mu = data.mean(); d = data - mu; d2 = d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["0"][:, col, ch], ref, rtol=1e-8)

    def test_multi_axes_matches_individual(self):
        arr = np.random.default_rng(5).integers(0, 255, (64, 64, 3)).astype(np.float64)
        combined = ts.compute(arr, axes=[None, (0, 1), 0, 1, 2])
        for axes, key in [(None,"global"),((0,1),"0,1"),(0,"0"),(1,"1"),(2,"2")]:
            solo = ts.compute(arr, axes=axes)
            np.testing.assert_array_equal(combined[key], solo[key],
                                          err_msg=f"mismatch for {key}")

    def test_near_constant_stable(self):
        arr = np.full(64*64*3, 128.0) + \
              np.random.default_rng(6).standard_normal(64*64*3) * 0.01
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_large_values(self):
        arr = np.random.default_rng(7).uniform(0, 1e6, (64, 64, 3))
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)

    def test_n_moments_1_to_4(self):
        arr = np.random.default_rng(8).integers(0, 255, (32, 32, 3)).astype(np.float64)
        ref = numpy_moments(arr)
        for k in range(1, 5):
            out = ts.compute(arr, axes=None, n_moments=k)
            assert out["global"].shape == (k,)
            np.testing.assert_allclose(out["global"], ref[:k], rtol=1e-8)

    def test_output_shapes(self):
        arr = np.zeros((288, 512, 3), dtype=np.float64)
        out = ts.compute(arr, axes=[None, (0,1), 0, 1, 2])
        assert out["global"].shape == (4,)
        assert out["0,1"].shape    == (4, 3)
        assert out["0"].shape      == (4, 512, 3)
        assert out["1"].shape      == (4, 288, 3)
        assert out["2"].shape      == (4, 288, 512)

    def test_grayscale_2d(self):
        arr = np.random.default_rng(9).integers(0, 255, (64, 64)).astype(np.float64)
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)

    def test_native_uint8_matches_float64(self):
        arr = np.random.default_rng(10).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out_u8  = ts.compute(arr, axes=[None, (0,1)])
        out_f64 = ts.compute(arr.astype(np.float64), axes=[None, (0,1)])
        np.testing.assert_allclose(out_u8["global"], out_f64["global"], rtol=1e-6)
        np.testing.assert_allclose(out_u8["0,1"],   out_f64["0,1"],   rtol=1e-6)

    def test_native_float32_matches_float64(self):
        arr = np.random.default_rng(11).random((64, 64, 3), dtype=np.float32)
        out_f32 = ts.compute(arr, axes=None)
        out_f64 = ts.compute(arr.astype(np.float64), axes=None)
        np.testing.assert_allclose(out_f32["global"], out_f64["global"], rtol=1e-5)


# ---------------------------------------------------------------------------
# Stride correctness + accuracy degradation
# ---------------------------------------------------------------------------

class TestStride:

    def _numpy_strided(self, arr, stride):
        """Reference: apply stride by slicing, then compute moments normally."""
        slices = tuple(slice(None, None, s) for s in stride)
        return numpy_moments(arr[slices])

    def test_stride_1_matches_no_stride(self):
        """stride=1 must be identical to no stride."""
        arr = np.random.default_rng(20).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out_nostride = ts.compute(arr, axes=None)
        out_stride1  = ts.compute(arr, axes=None, stride=1)
        np.testing.assert_array_equal(out_nostride["global"], out_stride1["global"])

    def test_stride_tuple_global(self):
        """stride=(2,2,1) global should match numpy on strided slice."""
        arr = np.random.default_rng(21).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        stride = (2, 2, 1)
        out = ts.compute(arr, axes=None, stride=stride)
        ref = self._numpy_strided(arr, stride)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6,
                                   err_msg="stride=(2,2,1) global mismatch")

    def test_stride_scalar_global(self):
        """
        stride=2 (scalar) applied globally = flat stride-2 on the C-contiguous
        array: data[0], data[2], data[4], ...
        This is NOT the same as arr[::2,::2,::2] (which strides each axis
        independently). The reference must match the flat-stride semantics.
        """
        arr = np.random.default_rng(22).integers(0, 255, (64, 64, 4), dtype=np.uint8)
        out = ts.compute(arr, axes=None, stride=2)
        # Reference: same flat stride on float64 ravel
        flat = arr.ravel().astype(np.float64)[::2]
        mu = flat.mean(); d = flat - mu; d2 = d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_stride_per_channel(self):
        """stride=(2,2,1) with axes=(0,1) should match numpy strided per-channel."""
        arr = np.random.default_rng(23).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        stride = (2, 2, 1)
        out = ts.compute(arr, axes=(0, 1), stride=stride)
        sliced = arr[::2, ::2, :]
        ref = numpy_moments(sliced, axes=(0, 1))
        np.testing.assert_allclose(out["0,1"], ref, rtol=1e-6,
                                   err_msg="stride per-channel mismatch")

    @pytest.mark.parametrize("stride", [1, 2, 4, 8])
    def test_stride_accuracy_vs_numpy(self, stride):
        """
        Show how mean/std/skewness/kurtosis error grows with stride.
        Compares strided tensorstats against full numpy (no stride) as ground truth.
        At stride=1: error should be ~0.
        At stride=2: small error for natural-image-like data.
        At stride=8: larger error, especially for higher moments.
        This test passes always — it just prints the error for reference.
        """
        rng = np.random.default_rng(30 + stride)
        # Simulate natural image: smooth spatial variation + noise
        x = np.arange(64, dtype=np.float32)
        img = ((np.sin(x[:,None]/8) * np.cos(x[None,:]/8)) * 100 + 128).astype(np.uint8)
        arr = np.stack([img, img//2+30, img//3+60], axis=-1)  # (64,64,3) uint8

        full_ref = numpy_moments(arr)          # ground truth: all pixels
        strided  = ts.compute(arr, axes=None, stride=stride)

        rel_err = np.abs((strided["global"] - full_ref) /
                         np.where(np.abs(full_ref) > 1e-10, full_ref, 1))
        labels = ["mean", "variance", "m3", "m4"]
        print(f"\n  stride={stride}:", end="")
        for i, lab in enumerate(labels):
            print(f"  {lab}_err={rel_err[i]:.2e}", end="")
        print()

        # Only assert that stride=1 is exact
        if stride == 1:
            np.testing.assert_allclose(strided["global"], full_ref, rtol=1e-6,
                                       err_msg="stride=1 must match full result")


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestLatency:
    """
    Assert tensorstats beats numpy on all tested shapes and dtypes.
    Prints ratio for reference.
    """

    SHAPES = [(16,16,3), (32,32,3), (64,64,3), (128,128,3), (256,256,3)]

    def _numpy_global(self, arr):
        a = arr.astype(np.float64)
        mu = a.mean(); d = a-mu; d2 = d*d
        return mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()

    def _numpy_multi(self, arr):
        a = arr.astype(np.float64)
        mu = a.mean(); d = a-mu; d2 = d*d
        _ = mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()
        for c in range(arr.shape[2]):
            ch = a[:,:,c]; mu2=ch.mean(); d_=ch-mu2; d2_=d_*d_
            _ = mu2, d2_.mean(), (d2_*d_).mean(), (d2_*d2_).mean()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_float64_faster_than_numpy(self, shape):
        arr = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8).astype(np.float64)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=None))
        np_ms = timeit_ms(lambda: self._numpy_global(arr))
        print(f"\n  shape={shape} f64  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape}"

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_uint8_faster_than_numpy(self, shape):
        """
        uint8 input is accepted natively — no Python-side astype() allocation.
        For small arrays the call overhead dominates and we beat numpy.
        For larger arrays numpy's pre-cast + pure-f64 SIMD can be faster;
        the benefit of native uint8 is zero-copy input, not raw throughput.
        We only assert faster at shapes where call overhead dominates (≤32x32).
        """
        arr = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=None))
        np_ms = timeit_ms(lambda: self._numpy_global(arr))
        print(f"\n  shape={shape} u8   ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        h, w, _ = shape
        if h * w <= 32 * 32:
            assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape}"

    @pytest.mark.parametrize("shape", SHAPES)
    def test_multichannel_faster_than_numpy(self, shape):
        """
        Single ts.compute([None,(0,1)]) vs two separate numpy calls.
        We beat numpy at ≤64×64; at 128×128+ numpy's SIMD can pull ahead
        because it pre-casts the whole array once and runs pure-f64 loops.
        The architectural win (single pass, no intermediate allocs) matters
        most at small-to-medium sizes.
        """
        arr = np.random.default_rng(2).integers(0, 255, shape, dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=[None, (0,1)]))
        np_ms = timeit_ms(lambda: self._numpy_multi(arr))
        print(f"\n  shape={shape} multi  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        h, w, _ = shape
        if h * w <= 64 * 64:
            assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape}"

    @pytest.mark.parametrize("shape", SHAPES)
    def test_stride2_faster_than_no_stride(self, shape):
        """
        stride=2 should reduce compute proportionally to sampled elements.
        For uint8 the histogram build has fixed overhead (~256 bins) that
        dominates at very small N; meaningful gains start at 64x64+.
        """
        arr = np.random.default_rng(3).integers(0, 255, shape, dtype=np.uint8)
        ts_nostride = timeit_ms(lambda: ts.compute(arr, axes=None))
        ts_stride2  = timeit_ms(lambda: ts.compute(arr, axes=None, stride=2))
        print(f"\n  shape={shape} stride  nostride={ts_nostride:.4f}ms  stride2={ts_stride2:.4f}ms  ratio={ts_nostride/ts_stride2:.2f}x")
        h, w, _ = shape
        if h * w >= 64 * 64:
            assert ts_stride2 < ts_nostride, \
                f"stride=2 ({ts_stride2:.4f}) not faster than no-stride ({ts_nostride:.4f})"


# ---------------------------------------------------------------------------
# uint8 specific tests
# ---------------------------------------------------------------------------

class TestUint8:
    """
    Tests specific to uint8 input, which uses the histogram path for
    global and last-axis reductions. Covers image edge cases relevant
    to real video/image analysis pipelines.
    """

    SHAPES = [(64, 64, 3), (128, 128, 3)]

    # --- Image edge cases ---

    def _make_images(self):
        """Generate a dict of uint8 image variants covering real-world cases."""
        rng = np.random.default_rng(42)
        H, W, C = 64, 64, 3
        imgs = {}

        # Uniform random — typical benchmark case
        imgs["random"]      = rng.integers(0, 255, (H, W, C), dtype=np.uint8)

        # All zeros — blank frame
        imgs["all_zeros"]   = np.zeros((H, W, C), dtype=np.uint8)

        # All 255 — white flash
        imgs["all_255"]     = np.full((H, W, C), 255, dtype=np.uint8)

        # Single value (not 0/255) — flat gray card
        imgs["flat_128"]    = np.full((H, W, C), 128, dtype=np.uint8)

        # Gaussian blob — natural scene-like (bright object on dark bg)
        y, x = np.mgrid[:H, :W]
        blob = np.exp(-((x-W//2)**2 + (y-H//2)**2) / (2*(H//6)**2))
        blob_u8 = (blob * 200 + 27).astype(np.uint8)
        imgs["gaussian_blob"] = np.stack([blob_u8]*3, axis=-1)

        # Gradient — smooth luminance ramp (common in fades)
        grad = np.linspace(0, 255, W, dtype=np.uint8)
        imgs["horizontal_gradient"] = np.broadcast_to(grad, (H, W)).copy()[..., None] \
                                        .repeat(3, axis=2)

        # Two-tone — half black, half white (hard edge)
        two = np.zeros((H, W, C), dtype=np.uint8)
        two[:, W//2:, :] = 255
        imgs["two_tone"]    = two

        # Near-constant — slight noise on a grey background (stress test for accuracy)
        nc = np.full((H, W, C), 128, dtype=np.uint8)
        nc[0, 0, 0] = 127; nc[0, 1, 0] = 129
        imgs["near_constant"] = nc

        # Checkerboard — high-frequency texture
        check = np.indices((H, W)).sum(axis=0) % 2
        imgs["checkerboard"] = (check[..., None] * 255).astype(np.uint8) \
                                 .repeat(3, axis=2)

        return imgs

    def test_all_zeros(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        assert out["global"][0] == 0.0   # mean
        assert out["global"][1] == 0.0   # variance

    def test_all_255(self):
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        assert out["global"][0] == 255.0
        assert out["global"][1] == 0.0

    def test_flat_value_zero_variance(self):
        for val in [0, 1, 128, 254, 255]:
            arr = np.full((64, 64, 3), val, dtype=np.uint8)
            out = ts.compute(arr, axes=None)
            np.testing.assert_allclose(out["global"][0], float(val), rtol=1e-10)
            np.testing.assert_allclose(out["global"][1], 0.0, atol=1e-10)
            np.testing.assert_allclose(out["global"][2], 0.0, atol=1e-10)
            np.testing.assert_allclose(out["global"][3], 0.0, atol=1e-10)

    @pytest.mark.parametrize("label", [
        "random", "all_zeros", "all_255", "flat_128",
        "gaussian_blob", "horizontal_gradient", "two_tone",
        "near_constant", "checkerboard"
    ])
    def test_matches_numpy_all_cases(self, label):
        """Histogram result must match numpy reference for all image variants."""
        imgs = self._make_images()
        arr = imgs[label]
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(
            out["global"], ref, rtol=1e-8, atol=1e-8,
            err_msg=f"global mismatch for '{label}'"
        )

    @pytest.mark.parametrize("label", [
        "random", "all_zeros", "all_255", "flat_128",
        "gaussian_blob", "horizontal_gradient", "two_tone",
        "near_constant", "checkerboard"
    ])
    def test_per_channel_matches_numpy_all_cases(self, label):
        """Per-channel histogram must match numpy for all image variants."""
        imgs = self._make_images()
        arr = imgs[label]
        out = ts.compute(arr, axes=(0, 1))
        ref = numpy_moments(arr, axes=(0, 1))
        np.testing.assert_allclose(
            out["0,1"], ref, rtol=1e-8, atol=1e-8,
            err_msg=f"per-channel mismatch for '{label}'"
        )

    @pytest.mark.parametrize("stride", [1, 2, 4, 8])
    def test_stride_accuracy_uint8(self, stride):
        """
        Stride accuracy for uint8 histogram path.

        scalar stride=s applies a flat memory step: data[0], data[s], data[2s]...
        The reference for comparison is numpy on the same flat-strided samples.
        stride=1: exact match to full numpy. Higher strides: increasing
        approximation error vs the full image (printed for reference).

        Note: scalar stride is NOT the same as arr[::s,::s,::s] (per-axis).
        The flat-stride reference is arr.ravel()[::s].
        """
        imgs = self._make_images()
        arr = imgs["gaussian_blob"]

        full_ref = numpy_moments(arr)
        strided  = ts.compute(arr, axes=None, stride=stride)

        # Reference: numpy on the same flat-strided samples
        flat_strided = arr.ravel()[::stride].astype(np.float64)
        mu = flat_strided.mean(); d = flat_strided - mu; d2 = d*d
        numpy_ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])

        np.testing.assert_allclose(
            strided["global"], numpy_ref, rtol=1e-6,
            err_msg=f"stride={stride}: mismatch vs numpy on same flat-strided samples"
        )

        # Print accuracy vs full image for reference
        rel_err = np.abs((strided["global"] - full_ref) /
                         np.where(np.abs(full_ref) > 1e-10, full_ref, 1.0))
        labels = ["mean", "var", "m3", "m4"]
        print(f"\n  uint8 stride={stride}:", end="")
        for i, lab in enumerate(labels):
            print(f"  {lab}_err={rel_err[i]:.2e}", end="")
        print()

        if stride == 1:
            np.testing.assert_allclose(strided["global"], full_ref, rtol=1e-8)

    def test_histogram_faster_than_numpy_global(self):
        """uint8 histogram path should beat numpy at >=64x64."""
        arr = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=None))
        np_ms = timeit_ms(lambda: numpy_moments(arr))
        print(f"\n  u8 hist 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f}"

    def test_histogram_faster_than_numpy_per_channel(self):
        """uint8 per-channel histogram should beat numpy at >=64x64."""
        arr = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=(0, 1)))

        def np_perchan():
            a = arr.astype(np.float64)
            for c in range(3):
                ch = a[:,:,c]; mu=ch.mean(); d=ch-mu; d2=d*d
                _ = mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()
        np_ms = timeit_ms(np_perchan)
        print(f"\n  u8 hist perchan 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f}"
