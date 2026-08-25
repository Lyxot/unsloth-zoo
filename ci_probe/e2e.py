import os
os.environ["UNSLOTH_MLX_INT8_BACKEND"] = "metal_mpp"
import mlx.core as mx
from unsloth_zoo.mlx.int8_prefill import registry
from unsloth_zoo.mlx.int8_prefill.backends import metal_mpp, portable
mx.random.seed(0)
class F:
    def __init__(s,n,k,bits,gs):
        w=(mx.random.normal((n,k))*0.02).astype(mx.bfloat16)
        p,sc,b=mx.quantize(w,group_size=gs,bits=bits,mode="affine")
        s.d={"weight":p,"scales":sc,"biases":b}; s.bits=bits; s.group_size=gs; s.mode="affine"
    def __contains__(s,k): return k in s.d
    def __getitem__(s,k): return s.d[k]
    def get(s,k,d=None): return s.d.get(k,d)
def rel(a,b):
    a=a.astype(mx.float32); b=b.astype(mx.float32)
    return (mx.abs(a-b).max()/mx.maximum(mx.abs(b).max(),1e-8)).item()
print(f"{'shape':>14} {'gs':>4} {'rows':>6} {'metal/mlx':>11} {'port/mlx':>11} {'metal/port':>11} {'requantLSB':>11}")
for (n,k) in [(2048,2048),(5120,1536),(1280,4096)]:
    for gs in (32,64,128):
        m=F(n,k,4,gs); registry.clear()
        ok,why=registry.register_module(m,"x")
        if not ok: print("skip", why); continue
        e=registry.get(m["weight"])
        try:
            a=metal_mpp.requantize_weight(e.w,e.scales,e.biases,e.ws,e.bits,e.group_size)
            b=portable.requantize_weight(e.w,e.scales,e.biases,e.ws,e.bits,e.group_size)
            mx.eval(a,b); rq=int(mx.abs(a.astype(mx.int32)-b.astype(mx.int32)).max().item())
        except Exception as ex:
            print(f"{n}x{k} gs={gs}: requant RAISED {type(ex).__name__}: {str(ex)[:200]}"); continue
        for rows in (512,549,2000):
            x=(mx.random.normal((rows,k))*0.5).astype(mx.bfloat16)
            try:
                gm=metal_mpp.matmul(x,e,out_dtype=mx.float32)
                gp=portable.matmul(x,e,out_dtype=mx.float32)
                rf=mx.quantized_matmul(x,e.w,e.scales,e.biases,True,e.group_size,e.bits,"affine").astype(mx.float32)
                mx.eval(gm,gp,rf)
            except Exception as ex:
                print(f"{n}x{k} gs={gs} rows={rows}: RAISED {type(ex).__name__}: {str(ex)[:200]}"); continue
            print(f"{n}x{k:>7} {gs:>4} {rows:>6} {rel(gm,rf):11.3e} {rel(gp,rf):11.3e} {rel(gm,gp):11.3e} {rq:11d}")
