#!/usr/bin/env python3
"""
Demonstration of Triton NPU Optimization Skill

This script demonstrates the optimization patterns from triton_demo examples
and shows how to use the skill templates.

Key optimizations demonstrated:
1. Grid adaptation from GPU to NPU
2. Task distribution across NPU cores
3. Accuracy validation pattern
"""

import torch
import triton
import triton.language as tl
import torch_npu
import triton.runtime.driver as driver

print("="*60)
print("Triton NPU Optimization Skill Demonstration")
print("="*60)

# ============================================================================
# PART 1: UNDERSTANDING THE OPTIMIZATION PATTERN
# ============================================================================

print("\n1. Understanding the Optimization Pattern")
print("-"*40)

print("\nFrom triton_demo/ori1 to triton_demo/new1:")
print("  Original (GPU style): grid = (NV, NK, N * H)")
print("  Optimized (NPU style): grid = (num_core,)")
print("  Where num_core = get_npu_properties()['num_vectorcore']")

print("\nKey changes:")
print("  1. Single program_id dimension instead of multiple")
print("  2. Task distribution loop: for task_id in tl.range(core_id, task_num, num_core)")
print("  3. Manual index computation from task_id")
print("  4. Added NPU-specific parameters")

# ============================================================================
# PART 2: NPU PROPERTIES FUNCTION
# ============================================================================

print("\n\n2. NPU Properties Function")
print("-"*40)

def get_npu_properties():
    """Get NPU device properties including core count"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

print("Function: get_npu_properties()")
print("Returns NPU device properties including:")
print("  - num_vectorcore: Number of vector cores")
print("  - num_aicore: Number of AI cores")
print("  - Other hardware specifications")

# ============================================================================
# PART 3: GRID ADAPTATION PATTERN
# ============================================================================

print("\n\n3. Grid Adaptation Pattern")
print("-"*40)

print("\nOriginal GPU grid calculation:")
print("""
# Original (from ori1)
B, T, H, K, V = 2, 4096, 32, 256, 128
BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
grid = (NV, NK, B * H)  # 3D logical grid
""")

print("\nOptimized NPU grid calculation:")
print("""
# Optimized (from new1)
B, T, H, K, V = 2, 4096, 32, 256, 128
BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

# Get NPU core count
num_core = get_npu_properties()["num_vectorcore"]

# Calculate task distribution parameters
task_num = NV * NK * B * H  # Total tasks
knh_step = NK * B * H       # Step for K dimension
nh_step = B * H             # Step for batch*head

grid = (num_core,)  # 1D physical grid
""")

# ============================================================================
# PART 4: KERNEL ENTRY POINT OPTIMIZATION
# ============================================================================

print("\n\n4. Kernel Entry Point Optimization")
print("-"*40)

print("\nOriginal GPU kernel entry (ori1):")
print("""
@triton.jit
def kernel(...):
    i_v = tl.program_id(0)  # V dimension
    i_k = tl.program_id(1)  # K dimension
    i_nh = tl.program_id(2)  # Batch*Head
    
    i_n = i_nh // H
    i_h = i_nh % H
    # ... rest of kernel
""")

print("\nOptimized NPU kernel entry (new1):")
print("""
@triton.jit
def kernel(...,
    knh_step: tl.constexpr,
    nh_step: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
    ...):
    
    core_id = tl.program_id(0)
    
    for task_id in tl.range(core_id, task_num, num_core):
        i_v = task_id // knh_step
        i_k = task_id % knh_step // nh_step
        i_nh = task_id % knh_step % nh_step
        
        i_n = i_nh // H
        i_h = i_nh % H
        # ... rest of kernel (same as original)
""")

# ============================================================================
# PART 5: ACCURACY VALIDATION PATTERN
# ============================================================================

print("\n\n5. Accuracy Validation Pattern")
print("-"*40)

print("\nBased on triton_demo/ori1 main function:")
print("""
def main():
    # Set up test parameters
    B, T, H, K, V = 2, 4096, 32, 256, 128
    q = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
    
    # Run reference implementation
    print("Reference implementation begin")
    output_ref, final_state_ref = original_function(q=q, k=k, v=v)
    
    # Run optimized implementation
    print("Optimized implementation begin")
    output_opt, final_state_opt = optimized_function(q=q, k=k, v=v)
    
    # Compare statistics
    print(f"Reference output shape: {output_ref.shape}")
    print(f"Optimized output shape: {output_opt.shape}")
    
    print(f"Reference min/max/mean: {output_ref.min():.6f}, "
          f"{output_ref.max():.6f}, {output_ref.mean():.6f}")
    print(f"Optimized min/max/mean: {output_opt.min():.6f}, "
          f"{output_opt.max():.6f}, {output_opt.mean():.6f}")
    
    # Calculate difference
    diff = torch.max(torch.abs(output_ref - output_opt))
    print(f'Maximum difference: {diff:.6e}')
    
    # Validate with tolerance
    torch.testing.assert_close(output_ref, output_opt, atol=1e-3, rtol=1e-3)
    print("Accuracy validation passed")
""")

# ============================================================================
# PART 6: USING THE SKILL TEMPLATES
# ============================================================================

print("\n\n6. Using the Skill Templates")
print("-"*40)

print("\nAvailable templates in triton-npu-optimizer skill:")
print("  1. optimize_template.py - Complete optimization pattern")
print("  2. validate_accuracy.py - Accuracy validation framework")
print("  3. test_complex_operator.py - Comprehensive testing")

print("\nUsage workflow:")
print("  1. Start with optimize_template.py as base")
print("  2. Implement your operator following the pattern")
print("  3. Use validate_accuracy.py for accuracy validation")
print("  4. Use test_complex_operator.py for comprehensive testing")
print("  5. Reference triton_demo examples for real-world patterns")

# ============================================================================
# PART 7: OPTIMIZATION CHECKLIST
# ============================================================================

print("\n\n7. Optimization Checklist")
print("-"*40)

checklist = [
    ("Grid Adaptation", [
        "□ Replace multi-dimensional grid with single core dimension",
        "□ Get NPU core count using get_npu_properties()",
        "□ Set grid = (num_core,)",
    ]),
    
    ("Task Distribution", [
        "□ Calculate total tasks = product of original grid dimensions",
        "□ Calculate step sizes for each original dimension",
        "□ Implement task distribution loop in kernel",
        "□ Reconstruct original indices from task_id",
    ]),
    
    ("Kernel Parameters", [
        "□ Add NPU-specific parameters: knh_step, nh_step, task_num, num_core",
        "□ Pass all necessary parameters from launch function",
    ]),
    
    ("Accuracy Validation", [
        "□ Implement main function comparing original vs optimized",
        "□ Print statistics for comparison",
        "□ Calculate maximum difference",
        "□ Validate with tolerance (atol=1e-3, rtol=1e-3)",
    ]),
]

for category, items in checklist:
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

# ============================================================================
# PART 8: SUMMARY
# ============================================================================

print("\n\n8. Summary")
print("-"*40)

print("\nThe triton-npu-optimizer skill provides:")
print("  ✅ Complete optimization patterns from triton_demo examples")
print("  ✅ Templates for grid adaptation and task distribution")
print("  ✅ Accuracy validation framework")
print("  ✅ Comprehensive testing templates")
print("  ✅ Step-by-step optimization checklist")

print("\nKey takeaways from triton_demo optimization:")
print("  1. NPU requires physical core binding, not logical grids")
print("  2. Task distribution is manual but follows predictable patterns")
print("  3. Accuracy must be validated after optimization")
print("  4. The optimization pattern is reusable for other operators")

print("\n" + "="*60)
print("Demonstration Complete")
print("="*60)
print("\nNext steps:")
print("  1. Review triton_demo/ori1 and triton_demo/new1")
print("  2. Use optimize_template.py for your operator")
print("  3. Follow the optimization checklist")
print("  4. Validate accuracy with provided templates")

if __name__ == "__main__":
    # This is a demonstration script, no actual execution needed
    pass