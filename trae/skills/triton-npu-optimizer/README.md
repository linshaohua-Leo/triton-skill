# Triton NPU Optimizer Skill

This skill provides tools and templates for optimizing Triton kernels for Ascend NPU, based on the optimization patterns demonstrated in the `triton_demo` examples.

## Overview

The skill helps automate the migration and optimization of GPU-style Triton kernels to NPU by implementing:

1. **Grid adaptation**: Converting logical GPU grids to NPU physical core binding
2. **Task distribution**: Efficiently distributing work across NPU cores
3. **Accuracy validation**: Comparing original and optimized implementations
4. **Comprehensive testing**: Ensuring correctness and performance

## Files

### Core Skill File
- `SKILL.md` - Main skill definition and documentation

### Templates
- `optimize_template.py` - Complete optimization pattern with step-by-step guide
- `validate_accuracy.py` - Accuracy validation for complex operators
- `test_complex_operator.py` - Comprehensive testing framework

## Key Optimization Patterns

### From GPU to NPU Grid Adaptation

**Original GPU Style (ori1):**
```python
grid = (NV, NK, N * H)
kernel[grid](...)
```

**Optimized NPU Style (new1):**
```python
num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
kernel[grid](..., task_num=NV*NK*N*H, num_core=num_core, ...)
```

### Task Distribution in Kernel

**GPU Kernel Entry:**
```python
i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
```

**NPU Kernel Entry with Task Distribution:**
```python
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    i_v = task_id // knh_step
    i_k = task_id % knh_step // nh_step
    i_nh = task_id % knh_step % nh_step
```

## Usage Examples

### 1. Basic Optimization Workflow

```python
# Use the optimization template
from optimize_template import main as optimize_main
optimize_main()  # Shows step-by-step optimization
```

### 2. Accuracy Validation

```python
# Use the validation template
from validate_accuracy import ComplexOperatorValidator

validator = ComplexOperatorValidator(atol=1e-3, rtol=1e-3)
# Define test cases and run validation
```

### 3. Comprehensive Testing

```python
# Use the testing framework
from test_complex_operator import ComplexOperatorTester

class MyOperatorTester(ComplexOperatorTester):
    def create_test_inputs(self, **kwargs):
        # Implement for your operator
        pass
    
    def run_operator(self, **kwargs):
        # Implement for your operator
        pass

tester = MyOperatorTester("my_operator")
tester.run_comprehensive_test_suite()
```

## Based on triton_demo Examples

This skill is based on the optimization patterns demonstrated in:
- `agent-skills/triton_demo/ori1` - Original fused_recurrent_fwd kernel (GPU style)
- `agent-skills/triton_demo/new1` - Optimized fused_recurrent_fwd kernel (NPU style)

Key improvements in new1:
1. Physical core binding instead of logical grid
2. Task distribution across NPU cores
3. Added NPU property query
4. Maintained accuracy with validation

## Integration with Existing Skills

This skill complements the existing `triton_auto_migration` skills by:
1. Providing specific optimization patterns for complex operators
2. Adding comprehensive testing frameworks
3. Focusing on accuracy validation
4. Offering reusable templates

## Best Practices

1. **Start with accuracy**: Always validate numerical correctness before optimizing performance
2. **Use the templates**: Follow the provided templates for consistent optimization
3. **Test comprehensively**: Use the testing framework to catch edge cases
4. **Profile performance**: Use NPU profiling tools to identify bottlenecks
5. **Document changes**: Keep track of optimizations for future reference

## When to Use This Skill

Invoke this skill when:
- Migrating complex Triton operators from GPU to NPU
- Optimizing NPU performance for existing Triton kernels
- Validating accuracy after optimization
- Implementing task distribution patterns
- Testing complex operators comprehensively