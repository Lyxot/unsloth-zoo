import mlx.core as mx, traceback
HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""
SRC = """
    constexpr int N = 128;
    constexpr int K = 256;
    constexpr int TM = 128;
    constexpr int TN = 128;
    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<8>> op;
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));
    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
    op.run(tA, tB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            if (m < M && n < N) out[size_t(m) * N + n] = cT[i];
        }
    }
"""
k = mx.fast.metal_kernel(name="probe_i8", input_names=["xq","wq","m_dim"], output_names=["out"], header=HEADER, source=SRC)
M=N=128; K=256
xq = ((mx.arange(M*K) % 251) - 125).reshape(M,K).astype(mx.int8)
wq = ((mx.arange(N*K) % 241) - 120).reshape(N,K).astype(mx.int8)
try:
    got = k(inputs=[xq,wq,mx.array([M],dtype=mx.int32)], grid=(N//128*32*8,(M+127)//128,1), threadgroup=(32*8,1,1), output_shapes=[(M,N)], output_dtypes=[mx.int32])[0]
    want = mx.matmul(xq.astype(mx.float32), wq.astype(mx.float32).T).astype(mx.int32)
    mx.eval(got, want)
    print("RAN. equal:", bool(mx.array_equal(got,want).item()))
    print("got[0,:4]", got[0,:4].tolist(), "want[0,:4]", want[0,:4].tolist())
except Exception as e:
    print("FAILED:", type(e).__name__)
    print(str(e)[:3000])
