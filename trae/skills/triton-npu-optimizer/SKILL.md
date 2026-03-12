---
name: "triton-npu-optimizer"
description: "Optimizes Triton kernels for Ascend NPU with grid adaptation and task distribution. Invoke when migrating GPU Triton kernels to NPU or optimizing complex operators for NPU hardware."
---

# Triton NPU Optimizer Skill

This skill helps optimize Triton kernels for Ascend NPU by adapting GPU-style grid definitions to NPU's physical core architecture and implementing efficient task distribution patterns.

## Overview

The skill automates the optimization pattern demonstrated in the `triton_demo` examples:
- **ori1**: Original GPU-style Triton kernel with logical grid dimensions
- **new1**: Optimized NPU version with physical core binding and task distribution

## When to Use This Skill

Invoke this skill when:
1. Migrating GPU Triton kernels to Ascend NPU
2. Optimizing complex Triton operators for NPU hardware
3. Encountering grid-related performance issues on NPU
4. Need to validate accuracy after optimization
5. Implementing task distribution across NPU physical cores

## Core Optimization Patterns

### 1. Grid Adaptation Pattern

**Original GPU Style:**
```python
grid = (NV, NK, N * H)
kernel[grid](...)
```

**Optimized NPU Style:**
```python
import torch_npu
import triton.runtime.driver as driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
```

### 2. Task Distribution Pattern

**GPU Kernel Entry:**
```python
i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
i_n, i_h = i_nh // H, i_nh % H
```

**NPU Kernel Entry with Task Distribution:**
```python
core_id = tl.program_id(0)
task_num = NV * NK * N * H
knh_step = NK * N * H
nh_step = N * H

for task_id in tl.range(core_id, task_num, num_core):
    i_v = task_id // knh_step
    i_k = task_id % knh_step // nh_step
    i_nh = task_id % knh_step % nh_step
    i_n = task_id % knh_step % nh_step // H
    i_h = task_id % knh_step % nh_step % H
```

### 3. Kernel Parameter Adaptation

**Additional NPU Parameters:**
```python
def kernel(...,
    knh_step: tl.constexpr,
    nh_step: tl.constexpr,
    N: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
    ...):
```

## Step-by-Step Optimization Guide

### Step 1: Analyze Original Kernel Structure

1. Identify grid dimensions: `grid = (dim1, dim2, dim3, ...)`
2. Identify program_id usage: `tl.program_id(0)`, `tl.program_id(1)`, etc.
3. Map program_id indices to logical dimensions

### Step 2: Calculate Task Distribution Parameters

```python
# Calculate total tasks
task_num = dim1 * dim2 * dim3 * ...  # Product of all grid dimensions

# Calculate step sizes for each dimension
# Example for 3D grid (dim1, dim2, dim3):
step_dim2_dim3 = dim2 * dim3
step_dim3 = dim3

# In kernel:
# task_id = core_id + i * num_core
# dim1_idx = task_id // step_dim2_dim3
# dim2_idx = (task_id % step_dim2_dim3) // step_dim3
# dim3_idx = task_id % step_dim3
```

### Step 3: Modify Kernel Entry Point

Replace direct program_id indexing with task distribution loop:

```python
# Before:
i0 = tl.program_id(0)
i1 = tl.program_id(1)
i2 = tl.program_id(2)

# After:
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    i0 = task_id // step_dim2_dim3
    i1 = (task_id % step_dim2_dim3) // step_dim3
    i2 = task_id % step_dim3
```

### Step 4: Update Kernel Launch Configuration

```python
# Before:
grid = (dim1, dim2, dim3)
kernel[grid](...)

# After:
num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
kernel[grid](
    ...,
    knh_step=step_dim2_dim3,
    nh_step=step_dim3,
    N=dim1,  # or appropriate mapping
    task_num=task_num,
    num_core=num_core,
)
```

## Accuracy Validation

### Reference Implementation Pattern

Based on the `ori1` main function, create accuracy validation:

```python
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
    
    # Compare results
    print(f"Reference output shape: {output_ref.shape}")
    print(f"Optimized output shape: {output_opt.shape}")
    
    print(f"Reference min/max/mean: {output_ref.min():.6f}, {output_ref.max():.6f}, {output_ref.mean():.6f}")
    print(f"Optimized min/max/mean: {output_opt.min():.6f}, {output_opt.max():.6f}, {output_opt.mean():.6f}")
    
    # Calculate difference
    diff = torch.max(torch.abs(output_ref - output_opt))
    print(f'Maximum difference: {diff:.6e}')
    
    # Validate with tolerance
    torch.testing.assert_close(output_ref, output_opt, atol=1e-3, rtol=1e-3)
    print("✓ Accuracy validation passed")
```

## Common Pitfalls and Solutions

### 1. Incorrect Task Mapping
**Problem**: Wrong calculation of step sizes leads to incorrect index computation
**Solution**: Double-check the relationship between grid dimensions and logical indices

### 2. Performance Degradation
**Problem**: Task distribution overhead reduces performance
**Solution**: Ensure task granularity is appropriate (not too fine-grained)

### 3. Memory Alignment Issues
**Problem**: NPU requires specific memory alignment
**Solution**: Check and ensure tensor dimensions meet NPU alignment requirements

### 4. Logical Operator Errors
**Problem**: Using `and`/`or` instead of `&`/`|` in mask operations
**Solution**: Replace all logical operators with bitwise operators in Triton kernels

### 5. Unnecessary Conditional Logic
**Problem**: Introducing additional conditional checks in NPU-optimized kernels
**Solution**: Maintain original parameter modification patterns instead of creating new variables

## Performance Optimization Techniques

### 1. Parameter Reassignment Pattern
When kernel parameters need to be modified in conditional branches, reassign them directly instead of creating new variables:

```python
# Original pattern (correct)
if IS_VARLEN:
    T = local_length  # Direct reassignment

# Avoid this pattern (incorrect)
if IS_VARLEN:
    T_local = local_length  # Creates new variable
    # Requires T_local if IS_VARLEN else T everywhere
```

### 2. Leveraging `do_not_specialize`
For parameters that may be modified, use the `do_not_specialize` decorator to prevent compiler specialization:

```python
@triton.jit(do_not_specialize=['T'])
def kernel(..., T, ...):
    # T can be safely modified
    if condition:
        T = new_value
```

### 3. Maintaining Computational Consistency
Ensure optimized kernels preserve:
- Identical mathematical operations
- Identical memory access patterns  
- Identical control flow structures

### 4. Avoiding Runtime Conditionals
Move compile-time determinable conditions to kernel parameters:

```python
# Pass as constexpr parameter
@triton.jit
def kernel(..., IS_VARLEN: tl.constexpr, ...):
    # Compiler can optimize based on IS_VARLEN value
```

## Template Files

The skill includes reference templates:

1. **Optimization Template**: `optimize_template.py` - Shows the complete optimization pattern
2. **Validation Template**: `validate_accuracy.py` - Shows accuracy comparison pattern
3. **Test Template**: `test_complex_operator.py` - Comprehensive testing for complex operators

## Best Practices

1. **Start Simple**: Begin with small problem sizes for validation
2. **Incremental Testing**: Test each optimization step independently
3. **Accuracy First**: Ensure numerical correctness before performance optimization
4. **Profile Performance**: Use NPU profiling tools to identify bottlenecks
5. **Document Changes**: Keep track of all modifications for future reference

## Related Skills

- Use `triton-migration-skills` for basic GPU to NPU migration patterns
- Use `triton-debug-tools` for debugging NPU kernel issues
- Use `triton-performance-analyzer` for performance optimization guidance

## Examples

See the `triton_demo` directory for complete examples:
- `ori1`: Original fused_recurrent_fwd kernel (GPU style)
- `new1`: Optimized fused_recurrent_fwd kernel (NPU style)

The optimization demonstrates:
1. Grid adaptation from 3D to 1D physical core grid
2. Task distribution across NPU cores
3. Accuracy validation with tolerance checking
4. Performance improvement through better NPU utilization