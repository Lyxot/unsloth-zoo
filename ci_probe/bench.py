import os, time
os.environ["UNSLOTH_MLX_INT8_BACKEND"] = "metal_mpp"
import mlx.core as mx
from unsloth_zoo.mlx.int8_prefill import registry
from unsloth_zoo.mlx.int8_prefill.backends import metal_mpp
mx.random.seed(0)
class F:
    def __init__(s,n,k,bits,gs):
        w=(mx.random.normal((n,k))*0.02).astype(mx.bfloat16)
        p,sc,b=mx.quantize(w,group_size=gs,bits=bits,mode="affine")
        s.d={"weight":p,"scales":sc,"biases":b}; s.bits=bits; s.group_size=gs; s.mode="affine"
    def __contains__(s,k): return k in s.d
    def __getitem__(s,k): return s.d[k]
    def get(s,k,d=None): return s.d.get(k,d)
def t(fn,n=20):
    for _ in range(5): mx.eval(fn())
    mx.synchronize(); a=time.perf_counter()
    for _ in range(n): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-a)/n*1000
print(f"{'N x K':>14} {'rows':>6} {'mlx_ms':>9} {'int8_ms':>9} {'speedup':>8}")
for (n,k) in [(4096,4096),(2048,2048)]:
    m=F(n,k,4,64); registry.clear()
    ok,why=registry.register_module(m,"x")
    if not ok: print("skip",why); continue
    e=registry.get(m["weight"])
    for rows in (512,2048,6400):
        x=(mx.random.normal((rows,k))*0.5).astype(mx.bfloat16)
        try:
            tr=t(lambda: mx.quantized_matmul(x,e.w,e.scales,e.biases,True,e.group_size,e.bits,"affine"))
            ti=t(lambda: metal_mpp.matmul(x,e,out_dtype=mx.bfloat16))
        except Exception as ex:
            print(f"{n}x{k} rows={rows}: RAISED {type(ex).__name__}: {str(ex)[:200]}"); continue
        print(f"{n}x{k:>7} {rows:>6} {tr:9.3f} {ti:9.3f} {tr/ti:7.2f}x")
