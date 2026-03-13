
import torch
import triton
import triton.language as tl
import torch_npu


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [4, 8]
#     ],
#     key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK', 'USE_GV'],
#     **autotune_cache_kwargs,
# )
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * tl.exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H*K
        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_o += (-1 if REVERSE else 1) * H*V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H*K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


# 融合循环前向计算
def fused_recurrent_fwd(
    q: torch.Tensor, # B, T, H, K
    k: torch.Tensor, # B, T, H, K
    v: torch.Tensor, # B, T, H, V
    g: torch.Tensor | None = None,  # B, T, H
    g_gamma: torch.Tensor | None = None,    # H
    gk: torch.Tensor | None = None, # B, T, H, K
    gv: torch.Tensor | None = None, # B, T, H, V
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # B, H, K, V
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
    )
    o = o.sum(0)
    return o, ht


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [4, 8]
#     ],
#     key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK', 'USE_GV'],
#     **autotune_cache_kwargs,
# )
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_fwd_kernel_new(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    k_off = (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    v_off = (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_q = q + k_off
    p_k = k + k_off
    p_v = v + v_off
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + ((T-1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + k_off
    if USE_GV:
        p_gv = gv + v_off
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * tl.exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        step_k = (-1 if REVERSE else 1) * H*K
        step_v = (-1 if REVERSE else 1) * H*V
        p_q += step_k
        p_k += step_k
        p_v += step_v
        p_o += step_v
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += step_k
        if USE_GV:
            p_gv += step_v

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


# B：Batch Size，批次大小，即一次处理的样本数（定长序列为样本数，变长序列为 cu_seqlens 推导的实际 batch 数）；
# T：Sequence Length，序列长度，即每个样本的 token 数量，也是代码中循环遍历的维度；
# H：Head Number，注意力头数，多注意力头并行计算，代码中按头独立处理（i_h 为头索引）；
# K：Head Dimension，单注意力头的维度，即 q/k 的最后一维特征数，是注意力打分的核心维度。

# B：batch size（批次大小，定长序列时的样本数）；
# T：sequence length（序列长度，单个样本的 token 数）；
# H：number of heads（注意力头数）；
# K：dim of q/k（每个 head 的 q/k 维度）；
# V：dim of v（每个 head 的 v 维度，通常 K=V）。

# 融合循环前向计算
def fused_recurrent_fwd_new(
    q: torch.Tensor, # B, T, H, K
    k: torch.Tensor, # B, T, H, K
    v: torch.Tensor, # B, T, H, V
    g: torch.Tensor | None = None,  # B, T, H
    g_gamma: torch.Tensor | None = None,    # H
    gk: torch.Tensor | None = None, # B, T, H, K
    gv: torch.Tensor | None = None, # B, T, H, V
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # B, H, K, V
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    fused_recurrent_fwd_kernel_new[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
    )
    o = o.sum(0)
    return o, ht

def main():
    B, T, H, K, V = 2, 4096, 32, 256, 128
    q = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
    g = torch.randn(B, T, H, device='npu', dtype=torch.float32)
    initial_state = torch.randn(B, H, K, V, device='npu', dtype=torch.float32)
    scale = 0.5

    # 调用函数
    print("fused_recurrent_fwd begin")
    output_ref, final_state_ref = fused_recurrent_fwd(
        q=q, k=k, v=v, scale=scale
    )

    print("fused_recurrent_fwd_new begin")
    output, final_state = fused_recurrent_fwd_new(
        q=q, k=k, v=v, scale=scale
    )

    print(f"output_ref shape: {output_ref.shape}")
    # print(f"Final state shape: {final_state.shape}")
    print(f"output_ref min: {output_ref.min()}")
    print(f"output_ref max: {output_ref.max()}")
    print(f"output_ref mean: {output_ref.mean()}")


    print(f"Output shape: {output.shape}")
    # print(f"Final state shape: {final_state.shape}")
    print(f"Output min: {output.min()}")
    print(f"Output max: {output.max()}")
    print(f"Output mean: {output.mean()}")

    print('output diff', torch.max(torch.abs(output-output_ref)))
    torch.testing.assert_close(output_ref, output,atol=1e-3,rtol=1e-3)





if __name__ == "__main__":
    main()