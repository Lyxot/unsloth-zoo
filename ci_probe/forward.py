import mlx.core as mx, mlx.nn as nn
from unsloth_zoo.mlx import int8_prefill as ip
mx.random.seed(0)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.up=nn.Linear(2048,4096,bias=False); s.down=nn.Linear(4096,2048,bias=False)
    def __call__(s,x): return s.down(nn.silu(s.up(x)))
m=MLP(); nn.quantize(m, group_size=64, bits=4)
x=(mx.random.normal((1024,2048))*0.5).astype(mx.bfloat16)
base=m(x); mx.eval(base)
print("is_supported:", ip.is_supported(), "|", ip.reason())
en=ip.enable(); print("enable():", en)
if en:
    print("warmup:", ip.warmup(m)); print("registered:", ip.registered())
    print("self_test:", ip.self_test())
after=m(x); mx.eval(after)
same=bool(mx.array_equal(base,after).item())
d=(mx.abs(after.astype(mx.float32)-base.astype(mx.float32)).max()/mx.maximum(mx.abs(base.astype(mx.float32)).max(),1e-8)).item()
print(f"forward identical: {same}   max_rel_change: {d:.4e}")
