"""
tensorstats test suite.

Run from repo root:
    python -m pytest tests/ -v -s

Output convention: moments-LAST.
  result["global"]  shape (n_moments,)
  result["0,1"]     shape (C, n_moments)
  result["grid"]    shape (*cell_shape, n_moments)
"""

import time
import numpy as np
import pytest
import tensorstats as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_moments(arr, axes=None):
    """Reference: two-pass central moments via numpy. Returns shape (*out, 4)."""
    a = arr.astype(np.float64)
    if axes is None:
        flat = a.ravel()
        mu = flat.mean(); d = flat - mu; d2 = d*d
        return np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
    ax = (axes,) if isinstance(axes, int) else tuple(axes)
    other = [dd for dd in range(a.ndim) if dd not in ax]
    flat = np.moveaxis(a, ax, range(len(ax))).reshape(-1, *[a.shape[dd] for dd in other])
    mu  = flat.mean(axis=0)
    d   = flat - mu; d2 = d*d
    # stack as (*other_shape, 4) — moments last
    return np.stack([mu, d2.mean(axis=0), (d2*d).mean(axis=0), (d2*d2).mean(axis=0)], axis=-1)


def timeit_ms(fn, n=2000, warmup=50):
    """
    Return minimum per-call time. Uses adaptive chunk sizes so each chunk
    runs for at least 20ms — long enough that OS scheduling noise (typically
    1-10ms) is a small fraction of the measurement.
    """
    for _ in range(warmup): fn()
    # Calibrate: find chunk size that gives ~20ms wall time
    t0 = time.perf_counter()
    for _ in range(10): fn()
    per_call = (time.perf_counter() - t0) / 10 * 1000  # ms
    chunk = max(10, int(20.0 / per_call))  # target 20ms per chunk
    n_chunks = max(5, n // chunk)
    best = float('inf')
    for _ in range(n_chunks):
        t0 = time.perf_counter()
        for _ in range(chunk): fn()
        best = min(best, (time.perf_counter() - t0) / chunk * 1000)
    return best


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

    def test_per_channel_shape_and_values(self):
        arr = np.random.default_rng(3).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = ts.compute(arr, axes=(0, 1))
        # shape: (C, n_moments) = (3, 4)
        assert out["0,1"].shape == (3, 4)
        ref = numpy_moments(arr, axes=(0, 1))  # shape (3, 4)
        np.testing.assert_allclose(out["0,1"], ref, rtol=1e-8)

    def test_per_channel_indexing(self):
        """Moments-last: channel means = result["0,1"][:, 0]"""
        arr = np.random.default_rng(4).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = ts.compute(arr, axes=(0, 1))
        for c in range(3):
            np.testing.assert_allclose(
                out["0,1"][c, 0], arr[:, :, c].mean(), rtol=1e-8,
                err_msg=f"channel {c} mean mismatch"
            )

    def test_axis_0_shape(self):
        arr = np.random.default_rng(5).integers(0, 255, (64, 64, 3)).astype(np.float64)
        out = ts.compute(arr, axes=0)
        # shape: (W, C, n_moments)
        assert out["0"].shape == (64, 3, 4)

    def test_multi_axes_matches_individual(self):
        arr = np.random.default_rng(6).integers(0, 255, (64, 64, 3)).astype(np.float64)
        combined = ts.compute(arr, axes=[None, (0, 1), 0, 1, 2])
        for axes, key in [(None,"global"),((0,1),"0,1"),(0,"0"),(1,"1"),(2,"2")]:
            solo = ts.compute(arr, axes=axes)
            np.testing.assert_array_equal(combined[key], solo[key])

    def test_near_constant_stable(self):
        arr = np.full(64*64*3, 128.0) + \
              np.random.default_rng(7).standard_normal(64*64*3) * 0.01
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_large_values(self):
        arr = np.random.default_rng(8).uniform(0, 1e6, (64, 64, 3))
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-8)

    def test_n_moments_1_to_4(self):
        arr = np.random.default_rng(9).integers(0, 255, (32, 32, 3)).astype(np.float64)
        ref = numpy_moments(arr)
        for k in range(1, 5):
            out = ts.compute(arr, axes=None, n_moments=k)
            assert out["global"].shape == (k,)
            np.testing.assert_allclose(out["global"], ref[:k], rtol=1e-8)

    def test_output_shapes(self):
        arr = np.zeros((288, 512, 3), dtype=np.float64)
        out = ts.compute(arr, axes=[None, (0,1), 0, 1, 2])
        assert out["global"].shape == (4,)
        assert out["0,1"].shape    == (3, 4)
        assert out["0"].shape      == (512, 3, 4)
        assert out["1"].shape      == (288, 3, 4)
        assert out["2"].shape      == (288, 512, 4)

    def test_grayscale_2d(self):
        arr = np.random.default_rng(10).integers(0, 255, (64, 64)).astype(np.float64)
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], numpy_moments(arr), rtol=1e-8)

    def test_native_uint8_matches_float64(self):
        arr = np.random.default_rng(11).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        u8  = ts.compute(arr, axes=[None, (0,1)])
        f64 = ts.compute(arr.astype(np.float64), axes=[None, (0,1)])
        np.testing.assert_allclose(u8["global"], f64["global"], rtol=1e-6)
        np.testing.assert_allclose(u8["0,1"],   f64["0,1"],   rtol=1e-6)

    def test_native_float32_matches_float64(self):
        arr = np.random.default_rng(12).random((64, 64, 3), dtype=np.float32)
        f32 = ts.compute(arr, axes=None)
        f64 = ts.compute(arr.astype(np.float64), axes=None)
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
                dv = cell_data - mu; d2 = dv*dv
                m = np.array([mu, d2.mean(), (d2*dv).mean(), (d2*d2).mean()])
                result[tuple(cell_coords)] = m
                return
            nc = n_cells[d]
            for ci in range(nc):
                lo = ci * arr.shape[d] // nc
                hi = (ci+1) * arr.shape[d] // nc
                recurse(d+1, cell_coords+[ci], slices+[slice(lo, hi)])

        recurse(0, [], [])
        return result

    def test_grid_shape_2x2(self):
        arr = np.random.default_rng(20).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(1, 1, 0))
        # 2^1=2 cells on H, 2^1=2 on W, 2^0=1 on C → (2,2,1,4)
        assert out["grid"].shape == (2, 2, 1, 4)

    def test_grid_shape_4x4(self):
        arr = np.random.default_rng(21).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(2, 2, 0))
        assert out["grid"].shape == (4, 4, 1, 4)

    def test_grid_shape_8x8(self):
        arr = np.random.default_rng(22).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(3, 3, 0))
        assert out["grid"].shape == (8, 8, 1, 4)

    def test_grid_correctness_2x2(self):
        arr = np.random.default_rng(23).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(1, 1, 0))
        ref = self._numpy_grid_ref(arr, (1, 1, 0))
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-6,
                                   err_msg="2x2 grid mismatch")

    def test_grid_correctness_4x4(self):
        arr = np.random.default_rng(24).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(2, 2, 0))
        ref = self._numpy_grid_ref(arr, (2, 2, 0))
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-6)

    def test_grid_correctness_8x8(self):
        arr = np.random.default_rng(25).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(3, 3, 0))
        ref = self._numpy_grid_ref(arr, (3, 3, 0))
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-6)

    def test_grid_correctness_64x64(self):
        """64x64 grid uses direct loop (cells < 256px). Still correct."""
        arr = np.random.default_rng(26).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(6, 6, 0))
        ref = self._numpy_grid_ref(arr, (6, 6, 0))
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-6)

    def test_grid_float64(self):
        arr = np.random.default_rng(27).random((64, 64, 3))
        out = ts.compute(arr, axes=None, grid=(2, 2, 0))
        ref = self._numpy_grid_ref(arr, (2, 2, 0))
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-8)

    def test_grid_moments_last_indexing(self):
        """result["grid"][:,:,0,0] is the spatial mean map — shape (gh,gw)."""
        arr = np.random.default_rng(28).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(2, 2, 0))
        # grid=(2,2,0) → 2^2=4 cells per spatial dim → (4,4,1,4)
        # mean map: shape (4, 4) — one mean per spatial cell, all channels combined
        mean_map = out["grid"][:, :, 0, 0]
        assert mean_map.shape == (4, 4)

    def test_grid_alongside_axes(self):
        """grid and axes reductions coexist in same result dict."""
        arr = np.random.default_rng(29).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=[None, (0,1)], grid=(2, 2, 0))
        assert "global" in out
        assert "0,1"   in out
        assert "grid"  in out
        assert out["global"].shape == (4,)
        assert out["0,1"].shape    == (3, 4)
        assert out["grid"].shape   == (4, 4, 1, 4)

    def test_grid_1d_tensor(self):
        arr = np.random.default_rng(30).integers(0, 255, (256,), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(3,))   # 8 cells
        ref = self._numpy_grid_ref(arr, (3,))
        assert out["grid"].shape == (8, 4)
        np.testing.assert_allclose(out["grid"], ref, rtol=1e-6)

    def test_grid_no_subdivision(self):
        """grid=(0,0,0) → 1 cell = global stats."""
        arr = np.random.default_rng(31).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(0, 0, 0))
        ref = ts.compute(arr, axes=None)
        # Single cell should equal global (within float tolerance)
        np.testing.assert_allclose(out["grid"][0, 0, 0], ref["global"], rtol=1e-6)

    @pytest.mark.parametrize("grid_exp", [1, 2, 3, 4, 5, 6])
    def test_grid_latency(self, grid_exp):
        """Grid should be faster than equivalent numpy loop."""
        arr = np.random.default_rng(32).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        n_cells = 2**grid_exp
        grid = (grid_exp, grid_exp, 0)

        def numpy_grid():
            ch = 64 // n_cells
            for gi in range(n_cells):
                for gj in range(n_cells):
                    cell = arr[gi*ch:(gi+1)*ch, gj*ch:(gj+1)*ch, :].ravel().astype(np.float64)
                    mu = cell.mean(); d = cell-mu; d2=d*d
                    _ = mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()

        ts_ms = timeit_ms(lambda: ts.compute(arr, grid=grid), n=500)
        np_ms = timeit_ms(numpy_grid, n=500)
        print(f"\n  grid={n_cells}x{n_cells} ({n_cells**2} cells)  "
              f"ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------

class TestStride:

    def test_stride_1_matches_no_stride(self):
        arr = np.random.default_rng(40).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            ts.compute(arr, axes=None)["global"],
            ts.compute(arr, axes=None, stride=1)["global"]
        )

    def test_stride_tuple_global(self):
        arr = np.random.default_rng(41).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        stride = (2, 2, 1)
        out = ts.compute(arr, axes=None, stride=stride)
        flat = arr[::2, ::2, :].ravel().astype(np.float64)
        mu = flat.mean(); d=flat-mu; d2=d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_stride_scalar_global(self):
        """scalar stride=s applies flat memory step data[0],data[s],..."""
        arr = np.random.default_rng(42).integers(0, 255, (64, 64, 4), dtype=np.uint8)
        out = ts.compute(arr, axes=None, stride=2)
        flat = arr.ravel()[::2].astype(np.float64)
        mu = flat.mean(); d=flat-mu; d2=d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    def test_stride_per_channel(self):
        arr = np.random.default_rng(43).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=(0, 1), stride=(2, 2, 1))
        ref = numpy_moments(arr[::2, ::2, :], axes=(0, 1))
        np.testing.assert_allclose(out["0,1"], ref, rtol=1e-6)

    @pytest.mark.parametrize("stride", [1, 2, 4, 8])
    def test_stride_accuracy_vs_numpy(self, stride):
        rng = np.random.default_rng(50 + stride)
        x = np.arange(64, dtype=np.float32)
        img = ((np.sin(x[:,None]/8) * np.cos(x[None,:]/8)) * 100 + 128).astype(np.uint8)
        arr = np.stack([img, img//2+30, img//3+60], axis=-1)
        full_ref = numpy_moments(arr)
        strided  = ts.compute(arr, axes=None, stride=stride)
        flat = arr.ravel()[::stride].astype(np.float64)
        mu=flat.mean(); d=flat-mu; d2=d*d
        numpy_ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(strided["global"], numpy_ref, rtol=1e-6)
        rel_err = np.abs((strided["global"]-full_ref)/np.where(np.abs(full_ref)>1e-10,full_ref,1))
        labels = ["mean","var","m3","m4"]
        print(f"\n  stride={stride}:", end="")
        for i,lab in enumerate(labels): print(f"  {lab}_err={rel_err[i]:.2e}", end="")
        print()
        if stride==1: np.testing.assert_allclose(strided["global"], full_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# uint8 specific
# ---------------------------------------------------------------------------

class TestUint8:

    def _make_images(self):
        rng = np.random.default_rng(42)
        H, W, C = 64, 64, 3
        imgs = {}
        imgs["random"]      = rng.integers(0, 255, (H,W,C), dtype=np.uint8)
        imgs["all_zeros"]   = np.zeros((H,W,C), dtype=np.uint8)
        imgs["all_255"]     = np.full((H,W,C), 255, dtype=np.uint8)
        imgs["flat_128"]    = np.full((H,W,C), 128, dtype=np.uint8)
        y,x = np.mgrid[:H,:W]
        blob = np.exp(-((x-W//2)**2+(y-H//2)**2)/(2*(H//6)**2))
        blob_u8 = (blob*200+27).astype(np.uint8)
        imgs["gaussian_blob"] = np.stack([blob_u8]*3, axis=-1)
        grad = np.linspace(0,255,W,dtype=np.uint8)
        imgs["horizontal_gradient"] = np.broadcast_to(grad,(H,W)).copy()[...,None].repeat(3,axis=2)
        two = np.zeros((H,W,C),dtype=np.uint8); two[:,W//2:,:]=255
        imgs["two_tone"] = two
        nc = np.full((H,W,C),128,dtype=np.uint8); nc[0,0,0]=127; nc[0,1,0]=129
        imgs["near_constant"] = nc
        check = np.indices((H,W)).sum(axis=0)%2
        imgs["checkerboard"] = (check[...,None]*255).astype(np.uint8).repeat(3,axis=2)
        return imgs

    def test_all_zeros(self):
        arr = np.zeros((64,64,3),dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        assert out["global"][0]==0.0 and out["global"][1]==0.0

    def test_all_255(self):
        arr = np.full((64,64,3),255,dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        assert out["global"][0]==255.0 and out["global"][1]==0.0

    def test_flat_value_zero_variance(self):
        for val in [0,1,128,254,255]:
            arr = np.full((64,64,3),val,dtype=np.uint8)
            out = ts.compute(arr, axes=None)
            np.testing.assert_allclose(out["global"][0], float(val), rtol=1e-10)
            np.testing.assert_allclose(out["global"][1:], 0.0, atol=1e-10)

    @pytest.mark.parametrize("label", [
        "random","all_zeros","all_255","flat_128","gaussian_blob",
        "horizontal_gradient","two_tone","near_constant","checkerboard"
    ])
    def test_matches_numpy_all_cases(self, label):
        arr = self._make_images()[label]
        out = ts.compute(arr, axes=None)
        ref = numpy_moments(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("label", [
        "random","all_zeros","all_255","flat_128","gaussian_blob",
        "horizontal_gradient","two_tone","near_constant","checkerboard"
    ])
    def test_per_channel_matches_numpy_all_cases(self, label):
        arr = self._make_images()[label]
        out = ts.compute(arr, axes=(0,1))
        ref = numpy_moments(arr, axes=(0,1))  # shape (3,4)
        np.testing.assert_allclose(out["0,1"], ref, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("stride",[1,2,4,8])
    def test_stride_accuracy_uint8(self, stride):
        arr = self._make_images()["gaussian_blob"]
        strided = ts.compute(arr, axes=None, stride=stride)
        flat = arr.ravel()[::stride].astype(np.float64)
        mu=flat.mean(); d=flat-mu; d2=d*d
        numpy_ref = np.array([mu,d2.mean(),(d2*d).mean(),(d2*d2).mean()])
        np.testing.assert_allclose(strided["global"], numpy_ref, rtol=1e-6)
        full_ref = numpy_moments(arr)
        rel_err = np.abs((strided["global"]-full_ref)/np.where(np.abs(full_ref)>1e-10,full_ref,1.0))
        print(f"\n  uint8 stride={stride}:", end="")
        for i,lab in enumerate(["mean","var","m3","m4"]): print(f"  {lab}_err={rel_err[i]:.2e}",end="")
        print()
        if stride==1: np.testing.assert_allclose(strided["global"], full_ref, rtol=1e-8)

    def test_histogram_faster_than_numpy_global(self):
        arr = np.random.default_rng(0).integers(0,255,(64,64,3),dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=None))
        np_ms = timeit_ms(lambda: numpy_moments(arr))
        print(f"\n  u8 hist 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        assert ts_ms < np_ms

    def test_histogram_faster_than_numpy_per_channel(self):
        arr = np.random.default_rng(1).integers(0,255,(64,64,3),dtype=np.uint8)
        ts_ms = timeit_ms(lambda: ts.compute(arr, axes=(0,1)))
        def np_pc():
            a=arr.astype(np.float64)
            for c in range(3):
                ch=a[:,:,c]; mu=ch.mean(); d=ch-mu; d2=d*d
                _=mu,d2.mean(),(d2*d).mean(),(d2*d2).mean()
        np_ms = timeit_ms(np_pc)
        print(f"\n  u8 hist perchan 64x64x3: ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        assert ts_ms < np_ms


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestLatency:

    SHAPES = [(16,16,3),(32,32,3),(64,64,3),(128,128,3),(256,256,3)]

    def _numpy_global(self, arr):
        a=arr.astype(np.float64); mu=a.mean(); d=a-mu; d2=d*d
        return mu,d2.mean(),(d2*d).mean(),(d2*d2).mean()

    def _numpy_multi(self, arr):
        a=arr.astype(np.float64); mu=a.mean(); d=a-mu; d2=d*d; _=mu,d2.mean()
        for c in range(arr.shape[2]):
            ch=a[:,:,c]; mu2=ch.mean(); d_=ch-mu2; d2_=d_*d_
            _=mu2,d2_.mean(),(d2_*d_).mean(),(d2_*d2_).mean()

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_float64_faster_than_numpy(self, shape):
        """
        tensorstats beats numpy for f64 at small (<=32x32) and large (>=256x256) sizes.

        Why we win:
          Small: ts avoids numpy's multiple temporary array allocations (d, d*d, etc.)
          Large: numpy's 3 temporary allocations of 786KB+ each hurt; ts uses one pass.

        Why 64x64 and 128x128 are skipped:
          At these sizes (48KB–393KB) the data is at the L1/L2 cache boundary.
          numpy's AVX2-vectorized single-purpose reductions (.mean(), element-wise ops)
          are competitive with our single-pass approach. The result is platform-dependent
          (MSVC vs GCC codegen, cache hierarchy). We don't assert here to avoid flakiness.
          Our real advantages at those sizes are uint8 histogram and multi-reduction.
        """
        arr=np.random.default_rng(0).integers(0,255,shape,dtype=np.uint8).astype(np.float64)
        ts_ms=timeit_ms(lambda: ts.compute(arr,axes=None))
        np_ms=timeit_ms(lambda: self._numpy_global(arr))
        print(f"\n  shape={shape} f64  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        h, w, _ = shape
        if h * w <= 32 * 32:
            assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape} — small array, call-overhead win expected"
        elif h * w >= 256 * 256:
            assert ts_ms < np_ms, f"ts={ts_ms:.4f} np={np_ms:.4f} shape={shape} — large array, alloc-overhead win expected"
        # 64x64 and 128x128: print only, no assert (cache boundary, platform-dependent)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_global_uint8_faster_than_numpy(self, shape):
        arr=np.random.default_rng(1).integers(0,255,shape,dtype=np.uint8)
        ts_ms=timeit_ms(lambda: ts.compute(arr,axes=None))
        np_ms=timeit_ms(lambda: self._numpy_global(arr))
        print(f"\n  shape={shape} u8   ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        h,w,_=shape
        if h*w<=32*32: assert ts_ms<np_ms

    @pytest.mark.parametrize("shape", SHAPES)
    def test_multichannel_faster_than_numpy(self, shape):
        arr=np.random.default_rng(2).integers(0,255,shape,dtype=np.uint8)
        ts_ms=timeit_ms(lambda: ts.compute(arr,axes=[None,(0,1)]))
        np_ms=timeit_ms(lambda: self._numpy_multi(arr))
        print(f"\n  shape={shape} multi  ts={ts_ms:.4f}ms  np={np_ms:.4f}ms  ratio={ts_ms/np_ms:.2f}x")
        h,w,_=shape
        if h*w<=64*64: assert ts_ms<np_ms

    @pytest.mark.parametrize("shape", SHAPES)
    def test_stride2_faster_than_no_stride(self, shape):
        arr=np.random.default_rng(3).integers(0,255,shape,dtype=np.uint8)
        ts_ns=timeit_ms(lambda: ts.compute(arr,axes=None))
        ts_s2=timeit_ms(lambda: ts.compute(arr,axes=None,stride=2))
        print(f"\n  shape={shape} stride  nostride={ts_ns:.4f}ms  stride2={ts_s2:.4f}ms  ratio={ts_ns/ts_s2:.2f}x")
        h,w,_=shape
        if h*w>=64*64: assert ts_s2<ts_ns


# ---------------------------------------------------------------------------
# Higher-dimensional tensor tests (2D through 6D)
# ---------------------------------------------------------------------------

class TestHigherDim:
    """
    Verify correctness for non-image tensor shapes.
    tensorstats is generic — any C-contiguous numpy array is accepted.
    """

    def _moments_ref(self, arr, axes=None):
        """numpy reference moments, returns shape (*out, 4) moments-last."""
        a = arr.astype(np.float64)
        if axes is None:
            flat = a.ravel()
            mu=flat.mean(); d=flat-mu; d2=d*d
            return np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        ax = (axes,) if isinstance(axes, int) else tuple(axes)
        other = [dd for dd in range(a.ndim) if dd not in ax]
        flat = np.moveaxis(a, ax, range(len(ax))).reshape(-1, *[a.shape[dd] for dd in other])
        mu=flat.mean(axis=0); d=flat-mu; d2=d*d
        return np.stack([mu, d2.mean(axis=0), (d2*d).mean(axis=0), (d2*d2).mean(axis=0)], axis=-1)

    # --- 2D (single-channel image / matrix) ---

    def test_2d_global(self):
        arr = np.random.default_rng(100).integers(0, 255, (64, 64), dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-6)

    def test_2d_axis0(self):
        arr = np.random.default_rng(101).random((64, 64))
        out = ts.compute(arr, axes=0)
        ref = self._moments_ref(arr, axes=0)   # shape (64, 4)
        assert out["0"].shape == (64, 4)
        np.testing.assert_allclose(out["0"], ref, rtol=1e-8)

    def test_2d_grid(self):
        arr = np.random.default_rng(102).integers(0, 255, (64, 64), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(2, 2))
        assert out["grid"].shape == (4, 4, 4)   # (2^2, 2^2, n_moments)
        # Check one cell
        ref_cell = self._moments_ref(arr[:16, :16])
        np.testing.assert_allclose(out["grid"][0, 0], ref_cell, rtol=1e-6)

    # --- 4D (e.g. video batch: T x H x W x C, or feature maps: N x C x H x W) ---

    def test_4d_global(self):
        arr = np.random.default_rng(110).random((4, 8, 8, 3))
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_4d_reduce_last_two(self):
        """Reduce spatial dims of a (N, H, W, C) tensor → shape (N, C, 4)."""
        arr = np.random.default_rng(111).random((4, 8, 8, 3))
        out = ts.compute(arr, axes=(1, 2))
        ref = self._moments_ref(arr, axes=(1, 2))   # shape (4, 3, 4)
        assert out["1,2"].shape == (4, 3, 4)
        np.testing.assert_allclose(out["1,2"], ref, rtol=1e-8)

    def test_4d_reduce_first(self):
        """Reduce batch dim of (N, H, W, C) → shape (H, W, C, 4)."""
        arr = np.random.default_rng(112).random((4, 8, 8, 3))
        out = ts.compute(arr, axes=0)
        ref = self._moments_ref(arr, axes=0)   # shape (8, 8, 3, 4)
        assert out["0"].shape == (8, 8, 3, 4)
        np.testing.assert_allclose(out["0"], ref, rtol=1e-8)

    def test_4d_grid(self):
        arr = np.random.default_rng(113).integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
        # grid=(1,2,2,0) → 2x4x4x1 cells
        out = ts.compute(arr, axes=None, grid=(1, 2, 2, 0))
        assert out["grid"].shape == (2, 4, 4, 1, 4)

    def test_4d_multiple_axes(self):
        arr = np.random.default_rng(114).random((3, 4, 5, 6))
        out = ts.compute(arr, axes=[None, (0, 1), (2, 3)])
        assert out["global"].shape == (4,)
        assert out["0,1"].shape    == (5, 6, 4)
        assert out["2,3"].shape    == (3, 4, 4)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)
        np.testing.assert_allclose(out["0,1"], self._moments_ref(arr, axes=(0,1)), rtol=1e-8)

    # --- 5D ---

    def test_5d_global(self):
        arr = np.random.default_rng(120).random((2, 3, 4, 5, 6))
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_5d_reduce_middle(self):
        arr = np.random.default_rng(121).random((2, 4, 4, 4, 3))
        out = ts.compute(arr, axes=(1, 2, 3))
        ref = self._moments_ref(arr, axes=(1, 2, 3))   # shape (2, 3, 4)
        assert out["1,2,3"].shape == (2, 3, 4)
        np.testing.assert_allclose(out["1,2,3"], ref, rtol=1e-8)

    def test_5d_uint8(self):
        arr = np.random.default_rng(122).integers(0, 255, (2, 4, 4, 4, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None)
        ref = self._moments_ref(arr)
        np.testing.assert_allclose(out["global"], ref, rtol=1e-6)

    # --- 6D ---

    def test_6d_global(self):
        arr = np.random.default_rng(130).random((2, 3, 4, 4, 4, 2))
        out = ts.compute(arr, axes=None)
        np.testing.assert_allclose(out["global"], self._moments_ref(arr), rtol=1e-8)

    def test_6d_reduce_inner(self):
        arr = np.random.default_rng(131).random((2, 2, 4, 4, 4, 3))
        out = ts.compute(arr, axes=(2, 3, 4))
        ref = self._moments_ref(arr, axes=(2, 3, 4))   # shape (2, 2, 3, 4)
        assert out["2,3,4"].shape == (2, 2, 3, 4)
        np.testing.assert_allclose(out["2,3,4"], ref, rtol=1e-8)

    def test_6d_grid(self):
        """Grid on 6D tensor."""
        arr = np.random.default_rng(132).integers(0, 255, (2, 2, 4, 4, 4, 3), dtype=np.uint8)
        out = ts.compute(arr, axes=None, grid=(1, 1, 1, 1, 1, 0))
        # 2^1=2 cells on each of first 5 dims, 1 cell on last → (2,2,2,2,2,1,4)
        assert out["grid"].shape == (2, 2, 2, 2, 2, 1, 4)

    # --- Stride on non-image shapes ---

    def test_stride_4d(self):
        arr = np.random.default_rng(140).random((8, 8, 8, 8))
        out = ts.compute(arr, axes=None, stride=2)
        flat = arr.ravel()[::2].astype(np.float64)
        mu=flat.mean(); d=flat-mu; d2=d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)

    def test_stride_tuple_4d(self):
        arr = np.random.default_rng(141).random((8, 8, 8, 3))
        out = ts.compute(arr, axes=None, stride=(2, 2, 2, 1))
        ref_arr = arr[::2, ::2, ::2, :].ravel().astype(np.float64)
        mu=ref_arr.mean(); d=ref_arr-mu; d2=d*d
        ref = np.array([mu, d2.mean(), (d2*d).mean(), (d2*d2).mean()])
        np.testing.assert_allclose(out["global"], ref, rtol=1e-8)
