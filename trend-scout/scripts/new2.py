# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl
import torch_npu
import triton.runtime.driver as driver

from fla.ops.utils import prepare_chunk_indices

def get_npu_properties():
    """Get NPU device properties including core count"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_npu(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    # NPU-specific parameters for task distribution
    nt_bh_step: tl.constexpr,  # Step for NT*B*H dimensions: NT * B * H
    bh_step: tl.constexpr,     # Step for B*H dimension: B * H
    task_num: tl.constexpr,    # Total number of tasks
    num_core: tl.constexpr,    # Number of NPU cores
):
    # NPU-style: single program_id for core index
    core_id = tl.program_id(0)
    
    # Distribute tasks across NPU cores
    for task_id in tl.range(core_id, task_num, num_core):
        # Reconstruct original indices from task_id
        # Original grid dimensions: (NV, NT, B*H)
        # where NV = triton.cdiv(V, BV)
        #       NT = triton.cdiv(T, BT) or len(chunk_indices)
        #       B*H = batch * heads
        
        # task_id = i_v * (NT * B * H) + i_t * (B * H) + i_bh
        # where i_bh = i_b * H + i_h
        
        i_v = task_id // nt_bh_step  # V dimension index
        remainder = task_id % nt_bh_step
        i_t = remainder // bh_step   # T dimension index
        i_bh = remainder % bh_step   # B*H dimension index
        
        i_b = i_bh // H
        i_h = i_bh % H

        if IS_VARLEN:
            i_tg = i_t
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            NT = tl.cdiv(T, BT)
        else:
            NT = tl.cdiv(T, BT)
            i_tg = i_b * NT + i_t
            bos, eos = i_b * T, i_b * T + T

        # offset calculation
        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        v_ptr = v + (bos * H + i_h) * V
        o_ptr = o + (bos * H + i_h) * V
        h_ptr = h + (i_tg * H + i_h).to(tl.int64) * K*V

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_ptr, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_ptr, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h_ptr, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))

            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)

        if USE_G:
            g_ptr = g + bos * H + i_h
            p_g = tl.make_block_ptr(g_ptr, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)
            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(v_ptr, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o_ptr, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        # to fix mma -> mma layout conversion
        # already solved by triton v3.2 or higher
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_o_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    
    # Get NPU core count
    num_core = get_npu_properties()["num_vectorcore"]
    
    # Calculate task distribution parameters
    # Original grid dimensions: (NV, NT, B * H)
    # where NV = triton.cdiv(V, meta['BV'])
    # We need to use the maximum BV from autotune configs for calculation
    max_BV = 128  # From autotune configs
    NV = triton.cdiv(V, max_BV)
    
    # Calculate step sizes for task distribution
    # task_id = i_v * (NT * B * H) + i_t * (B * H) + i_bh
    # where i_bh = i_b * H + i_h
    
    nt_bh_step = NT * B * H  # Step for NT*B*H dimensions
    bh_step = B * H          # Step for B*H dimension
    
    # Total number of tasks
    task_num = NV * NT * B * H
    
    # NPU grid: single dimension = number of cores
    grid = (num_core,)
    
    chunk_fwd_kernel_o_npu[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        nt_bh_step=nt_bh_step,
        bh_step=bh_step,
        task_num=task_num,
        num_core=num_core,
    )
    return o

def validate_accuracy():
    """Validate accuracy between original and optimized implementations"""
    print("=" * 60)
    print("Accuracy Validation for chunk_fwd_o")
    print("=" * 60)
    
    # Set up test parameters
    B, T, H, K, V = 2, 4096, 32, 256, 128
    BT = 64  # chunk_size
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    h = torch.randn(B * triton.cdiv(T, BT), H, K, V, dtype=torch.float32)
    
    # Move to NPU if available
    if torch.npu.is_available():
        q = q.npu()
        k = k.npu()
        v = v.npu()
        h = h.npu()
    
    # Run original implementation
    print("\n1. Running original implementation...")
    try:
        # Import original function
        from ori2 import chunk_fwd_o as original_chunk_fwd_o
        output_ref = original_chunk_fwd_o(
            q=q, k=k, v=v, h=h,
            g=None, g_gamma=None,
            scale=None,
            cu_seqlens=None,
            chunk_size=BT,
            chunk_indices=None
        )
        print(f"   Original output shape: {output_ref.shape}")
    except Exception as e:
        print(f"   Error running original: {e}")
        # Create dummy reference for demonstration
        output_ref = torch.randn(B, T, H, V, device=q.device, dtype=torch.float32)
    
    # Run optimized implementation
    print("\n2. Running optimized NPU implementation...")
    output_opt = chunk_fwd_o_npu(
        q=q, k=k, v=v, h=h,
        g=None, g_gamma=None,
        scale=None,
        cu_seqlens=None,
        chunk_size=BT,
        chunk_indices=None
    )
    print(f"   Optimized output shape: {output_opt.shape}")
    
    # Compare statistics
    print("\n3. Comparing statistics...")
    print(f"   Reference statistics:")
    print(f"     Min: {output_ref.min().item():.6f}")
    print(f"     Max: {output_ref.max().item():.6f}")
    print(f"     Mean: {output_ref.mean().item():.6f}")
    print(f"     Std: {output_ref.std().item():.6f}")
    
    print(f"\n   Optimized statistics:")
    print(f"     Min: {output_opt.min().item():.6f}")
    print(f"     Max: {output_opt.max().item():.6f}")
    print(f"     Mean: {output_opt.mean().item():.6f}")
    print(f"     Std: {output_opt.std().item():.6f}")
    
    # Calculate differences
    abs_diff = torch.abs(output_ref - output_opt)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    print(f"\n4. Difference analysis:")
    print(f"   Maximum absolute difference: {max_abs_diff:.6e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6e}")
    
    # Calculate relative difference
    ref_abs = torch.abs(output_ref)
    rel_diff = abs_diff / (ref_abs + 1e-8)  # Avoid division by zero
    max_rel_diff = rel_diff.max().item()
    
    print(f"   Maximum relative difference: {max_rel_diff:.6e}")
    
    # Validation with tolerance
    atol = 1e-3  # Absolute tolerance
    rtol = 1e-3  # Relative tolerance
    
    print(f"\n5. Validation with tolerance (atol={atol}, rtol={rtol}):")
    
    if max_abs_diff < atol:
        print("   ✓ Absolute difference within tolerance")
    else:
        print(f"   ✗ Absolute difference exceeds tolerance: {max_abs_diff:.6e} > {atol}")
    
    if max_rel_diff < rtol:
        print("   ✓ Relative difference within tolerance")
    else:
        print(f"   ✗ Relative difference exceeds tolerance: {max_rel_diff:.6e} > {rtol}")
    
    # Overall validation
    try:
        torch.testing.assert_close(
            output_ref, output_opt,
            atol=atol, rtol=rtol,
            msg="Accuracy validation failed"
        )
        print("\n✅ SUCCESS: Accuracy validation passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ FAILURE: Accuracy validation failed")
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    # Run validation if executed directly
    validate_accuracy()