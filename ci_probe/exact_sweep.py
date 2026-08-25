import os
os.environ["UNSLOTH_MLX_INT8_BACKEND"] = "metal_mpp"
import mlx.core as mx, numpy as np
from unsloth_zoo.mlx.int8_prefill.backends import metal_mpp
rng = np.random.default_rng(0); bad = 0; total = 0
for M in (1, 127, 128, 129, 512, 549, 1000, 2048):
    for N in (128, 256, 2048):
        for K in (32, 256, 1536, 4096):
            total += 1
            xq = rng.integers(-127,128,(M,K),dtype=np.int8); wq = rng.integers(-127,128,(N,K),dtype=np.int8)
            try:
                g = metal_mpp.int8_gemm_raw(mx.array(xq), mx.array(wq)); mx.eval(g)
            except Exception as e:
                bad += 1; print(f"M={M} N={N} K={K}: RAISED {type(e).__name__}: {str(e)[:200]}"); continue
            w = xq.astype(np.int32) @ wq.astype(np.int32).T
            if not np.array_equal(np.array(g), w):
                bad += 1; d = np.array(g)-w
                print(f"M={M} N={N} K={K}: MISMATCH wrong={np.count_nonzero(d)}/{d.size} max={np.abs(d).max()}")
print(f"RESULT bit-exact {total-bad}/{total}")
