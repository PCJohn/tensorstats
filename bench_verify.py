#!/usr/bin/env python3
"""tensorstats SIMD verify + benchmark. Run after `pip install -e .`.

Checks every code path against a numpy reference (quality) and measures
latency. Prints a summary block to paste back. No pytest needed.

    python bench_verify.py
"""
import platform
import sys
import time

import numpy as np

import tensorstats as ts


# ------------------------------- environment -------------------------------
def cpu_simd():
    flags = ""
    try:
        if platform.system() == "Linux":
            for ln in open("/proc/cpuinfo"):
                if ln.startswith("flags"):
                    flags = ln
                    break
        elif platform.system() == "Darwin":
            import subprocess

            flags = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features",
                 "machdep.cpu.leaf7_features"]
            ).decode().lower()
    except Exception:
        pass
    have = [x for x in ("avx512f", "avx2", "avx", "fma", "neon")
            if x in flags.lower()]
    if platform.machine().lower() in ("arm64", "aarch64"):
        have = have or ["neon"]
    return ",".join(have) or "unknown"


def env_block():
    print("=" * 60)
    print("ENV")
    print(f"  os        : {platform.system()} {platform.release()} "
          f"({platform.machine()})")
    print(f"  python    : {sys.version.split()[0]}")
    print(f"  numpy     : {np.__version__}")
    print(f"  cpu       : {platform.processor() or 'n/a'}")
    print(f"  simd      : {cpu_simd()}")
    print("=" * 60)


# ------------------------------- references --------------------------------
def ref_axes(a):  # per-channel over last axis
    C = a.shape[-1]
    f = a.reshape(-1, C).astype(np.float64)
    mu = f.mean(0)
    d = f - mu
    return np.stack([mu, (d ** 2).mean(0), (d ** 3).mean(0),
                     (d ** 4).mean(0)], 1)


def ref_global(a):
    f = a.reshape(-1).astype(np.float64)
    mu = f.mean()
    d = f - mu
    return np.array([mu, (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()])


def ref_grid(a, grid):
    nc = [1 << k for k in grid]
    ncell = int(np.prod(nc))
    cid = None
    for d in range(a.ndim):
        coord = (np.arange(a.shape[d]) * nc[d]) // a.shape[d]
        shp = [1] * a.ndim
        shp[d] = a.shape[d]
        c = coord.reshape(shp)
        cid = c if cid is None else cid * nc[d] + c
    cid = np.broadcast_to(cid, a.shape).reshape(-1)
    f = a.reshape(-1).astype(np.float64)
    cnt = np.maximum(np.bincount(cid, minlength=ncell), 1).astype(np.float64)
    mean = np.bincount(cid, weights=f, minlength=ncell) / cnt
    d = f - mean[cid]
    out = np.stack([mean] + [np.bincount(cid, weights=d ** k,
                   minlength=ncell) / cnt for k in (2, 3, 4)], 1)
    return out.reshape(*nc, 4)


def max_rel_err(out, ref):
    out, ref = np.asarray(out, float), np.asarray(ref, float)
    return float(np.abs((out - ref) / np.where(ref != 0, ref, 1)).max())


# ------------------------------- timing ------------------------------------
def bench(fn, warm=60, it=300, reps=15):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(it):
            fn()
        best = min(best, (time.perf_counter() - t) / it)
    return best * 1e3  # ms (p-min)


# ------------------------------- cases -------------------------------------
def make(shape, dt, rng):
    if dt == np.uint8:
        return rng.integers(0, 255, shape, dtype=np.uint8)
    return rng.random(shape).astype(dt)


def run():
    env_block()
    rng = np.random.default_rng(0)
    tol = {np.uint8: 1e-6, np.float64: 1e-8, np.float32: 1e-4}
    rows = []
    worst_err = 0.0
    fails = 0

    def do(label, shape, dt, axes, grid, ref):
        nonlocal worst_err, fails
        a = make(shape, dt, rng)
        sc = ts.StatsComputer(shape=shape, axes=axes, grid=grid)
        out = sc.compute(a)
        key = ("grid_0" if grid is not None
               else ("global" if axes == [None] else "0,1"))
        err = max_rel_err(np.array(out[key]), ref(a))
        ok = err <= tol[dt]
        worst_err = max(worst_err, err)
        fails += (0 if ok else 1)
        ms = bench(lambda: sc.compute(a))
        rows.append((label, np.dtype(dt).name, f"{ms:8.4f}",
                     f"{err:.1e}", "ok" if ok else "FAIL"))

    S1, S2 = (64, 64, 3), (288, 512, 3)
    # global
    for dt in (np.uint8, np.float64, np.float32):
        do("global 288x512x3", S2, dt, [None], None, ref_global)
    # per-channel, C scaling (channels-in-lanes)
    for dt in (np.float64, np.float32):
        for C in (3, 8, 16):
            do(f"per-chan 288x512x{C}", (288, 512, C), dt, [(0, 1)], None,
               ref_axes)
    do("per-chan 288x512x3", S2, np.uint8, [(0, 1)], None, ref_axes)
    # grid (4,4,0) = spatial, segmented path
    for dt in (np.uint8, np.float64):
        do("grid(4,4,0) 288x512x3", S2, dt, [], (4, 4, 0),
           lambda a: ref_grid(a, (4, 4, 0)))
        do("grid(4,4,0) 64x64x3", S1, dt, [], (4, 4, 0),
           lambda a: ref_grid(a, (4, 4, 0)))
    # grid (4,4,2) = channel-gridded, scalar fallback (control)
    do("grid(4,4,2) 288x512x3", S2, np.float64, [], (4, 4, 2),
       lambda a: ref_grid(a, (4, 4, 2)))
    # combined (framegate-like): global + per-channel + spatial grid
    a = make(S2, np.float64, rng)
    sc = ts.StatsComputer(shape=S2, axes=[None, (0, 1)], grid=(4, 4, 0))
    rows.append(("ALL f64 [g+pc+grid440]", "float64",
                 f"{bench(lambda: sc.compute(a)):8.4f}", "-", "-"))
    a8 = make(S2, np.uint8, rng)
    sc8 = ts.StatsComputer(shape=S2, axes=[None, (0, 1)], grid=(4, 4, 0))
    rows.append(("ALL u8 [g+pc+grid440]", "uint8",
                 f"{bench(lambda: sc8.compute(a8)):8.4f}", "-", "-"))

    print(f"{'case':26} {'dtype':8} {'p-min ms':>9}  {'rel_err':>8}  ok")
    print("-" * 60)
    for r in rows:
        print(f"{r[0]:26} {r[1]:8} {r[2]:>9}  {r[3]:>8}  {r[4]}")
    print("-" * 60)
    print(f"SUMMARY: quality {'PASS' if fails == 0 else f'FAIL({fails})'}  "
          f"worst_rel_err={worst_err:.2e}")
    print("(paste the ENV block + this table back)")


if __name__ == "__main__":
    run()
