# Agent Skills for Triton Migration and Optimization

This directory contains skills and templates for migrating and optimizing Triton operators from GPU to Ascend NPU.

## Skills Overview

### 1. Triton Auto Migration Skills
**Location**: `triton_auto_migration/`

Core migration skills for GPU to NPU adaptation:
- `triton_migration_skills.md` - Comprehensive migration guide with checklists
- `TRITON_AUTOMATED_MIGRATION_TO_ASCEND_NPU.md` - Detailed migration documentation
- `TRITON_TO_ASCEND_MIGRATION_SKILLS.md` - Quick reference skills
- `test_template.py` - Testing template for migrated operators
- `debug_template.py` - Debugging template for migration issues

**Key Topics**:
- Device adaptation (cuda → npu)
- Grid adjustment for NPU physical cores
- Logical operator conversion (and → &, or → |, not → ~)
- Memory alignment requirements
- Autotune configuration

### 2. Triton NPU Optimizer Skill (NEW)
**Location**: `.trae/skills/triton-npu-optimizer/`

Advanced optimization for complex Triton operators on NPU:
- `SKILL.md` - Skill definition and usage guidelines
- `optimize_template.py` - Complete optimization pattern
- `validate_accuracy.py` - Accuracy validation framework
- `test_complex_operator.py` - Comprehensive testing template

**Based on triton_demo examples**:
- `ori1` - Original fused_recurrent_fwd kernel (GPU style)
- `new1` - Optimized fused_recurrent_fwd kernel (NPU style)

**Key Optimization Patterns**:
- Grid adaptation from logical to physical cores
- Task distribution across NPU cores
- Accuracy validation with tolerance checking
- Performance benchmarking

### 3. Triton Demo Examples
**Location**: `triton_demo/`

Real-world examples of optimization:
- `ori1` - Original implementation with GPU-style grid
- `new1` - Optimized implementation with NPU task distribution

**Demonstrated Optimizations**:
1. **Grid adaptation**: `(NV, NK, N*H)` → `(num_core,)`
2. **Task distribution**: `tl.program_id()` → `tl.range(core_id, task_num, num_core)`
3. **NPU properties**: `get_npu_properties()["num_vectorcore"]`
4. **Accuracy validation**: Main function comparing original vs optimized

## Usage Workflow

### For Basic Migration:
1. Use `triton_auto_migration/triton_migration_skills.md` for step-by-step guidance
2. Apply device adaptation and operator conversion
3. Test with `test_template.py`
4. Debug with `debug_template.py` if needed

### For Complex Operator Optimization:
1. Use `.trae/skills/triton-npu-optimizer/optimize_template.py` as starting point
2. Implement grid adaptation and task distribution
3. Validate accuracy with `validate_accuracy.py`
4. Run comprehensive tests with `test_complex_operator.py`
5. Reference `triton_demo/` examples for patterns

## Key Concepts

### NPU vs GPU Architecture Differences

| Aspect | GPU | Ascend NPU |
|--------|-----|------------|
| **Grid concept** | Logical task dimensions | Physical core binding |
| **Core count** | Many SMs (streaming multiprocessors) | Fixed AI Cores |
| **Memory alignment** | Flexible | Strict (32B for VV, 512B for CV) |
| **Operator semantics** | Logical operators (and/or) | Bitwise operators (&/\|) |

### Optimization Patterns from triton_demo

**Original (ori1)**:
```python
grid = (NV, NK, N * H)
i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
```

**Optimized (new1)**:
```python
num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    i_v = task_id // knh_step
    i_k = task_id % knh_step // nh_step
    i_nh = task_id % knh_step % nh_step
```

## Best Practices

1. **Accuracy First**: Always validate numerical correctness before optimizing performance
2. **Incremental Testing**: Test each optimization step independently
3. **Use Templates**: Follow provided templates for consistency
4. **Profile Performance**: Use NPU profiling tools to identify bottlenecks
5. **Document Changes**: Keep track of optimizations for future reference

## Related Resources

- Triton Ascend documentation
- NPU performance profiling tools
- Triton community examples
- Migration case studies

## Getting Help

1. Check the migration skills for common issues
2. Reference the demo examples for optimization patterns
3. Use the testing templates for validation
4. Consult NPU documentation for hardware specifics