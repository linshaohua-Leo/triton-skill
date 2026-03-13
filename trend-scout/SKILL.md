---
name: "triton-npu-optimizer"
description: "为Ascend NPU优化Triton内核，实现网格适配和任务分发。在将GPU Triton内核迁移到NPU或为NPU硬件优化复杂算子时调用。"
---

# Triton NPU优化器技能

该技能通过将GPU风格的网格定义适配到NPU的物理核心架构并实现高效的任务分发模式，帮助优化Ascend NPU的Triton内核。

## 概述

该技能自动化了`triton_demo`示例中展示的优化模式：
- **ori1**: 具有逻辑网格维度的原始GPU风格Triton内核
- **new1**: 具有物理核心绑定和任务分发的优化NPU版本

## 使用场景

在以下场景中调用此技能：
1. 将GPU Triton内核迁移到Ascend NPU
2. 为NPU硬件优化复杂的Triton算子
3. 在NPU上遇到网格相关的性能问题
4. 需要在优化后验证准确性
5. 实现跨NPU物理核心的任务分发

## 核心优化模式

### 1. 网格适配模式

**原始GPU风格：**
```python
grid = (NV, NK, N * H)
kernel[grid](...)
```

**优化后的NPU风格：**
```python
import torch_npu
import triton.runtime.driver as driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
```

### 2. 任务分发模式

**GPU内核入口：**
```python
i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
i_n, i_h = i_nh // H, i_nh % H
```

**带任务分发的NPU内核入口：**
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

### 3. 内核参数适配

**额外的NPU参数：**
```python
def kernel(...,    
    knh_step: tl.constexpr,
    nh_step: tl.constexpr,
    N: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
    ...):
```

## 分步优化指南

### 步骤1：分析原始内核结构

1. 识别网格维度：`grid = (dim1, dim2, dim3, ...)`
2. 识别program_id用法：`tl.program_id(0)`, `tl.program_id(1)`等
3. 将program_id索引映射到逻辑维度

### 步骤2：计算任务分发参数

```python
# 计算总任务数
task_num = dim1 * dim2 * dim3 * ...  # 所有网格维度的乘积

# 计算每个维度的步长
# 3D网格(dim1, dim2, dim3)示例：
step_dim2_dim3 = dim2 * dim3
step_dim3 = dim3

# 在内核中：
# task_id = core_id + i * num_core
# dim1_idx = task_id // step_dim2_dim3
# dim2_idx = (task_id % step_dim2_dim3) // step_dim3
# dim3_idx = task_id % step_dim3
```

### 步骤3：修改内核入口点

用任务分发循环替换直接的program_id索引：

```python
# 之前：
i0 = tl.program_id(0)
i1 = tl.program_id(1)
i2 = tl.program_id(2)

# 之后：
core_id = tl.program_id(0)
for task_id in tl.range(core_id, task_num, num_core):
    i0 = task_id // step_dim2_dim3
    i1 = (task_id % step_dim2_dim3) // step_dim3
    i2 = task_id % step_dim3
```

### 步骤4：更新内核启动配置

```python
# 之前：
grid = (dim1, dim2, dim3)
kernel[grid](...)

# 之后：
num_core = get_npu_properties()["num_vectorcore"]
grid = (num_core,)
kernel[grid](
    ...,
    knh_step=step_dim2_dim3,
    nh_step=step_dim3,
    N=dim1,  # 或适当的映射
    task_num=task_num,
    num_core=num_core,
)
```

## 性能优化技术

### 1. 参数重赋值模式

当需要在内核参数的条件分支中修改参数时，直接重新赋值它们，而不是创建新变量：

```python
# 原始模式（正确）
if IS_VARLEN:
    T = local_length  # 直接重赋值

# 避免这种模式（错误）
if IS_VARLEN:
    T_local = local_length  # 创建新变量
    # 各处需要使用 T_local if IS_VARLEN else T
```

### 2. 利用 `do_not_specialize`

对于可能被修改的参数，使用 `do_not_specialize` 装饰器来防止编译器特化：

```python
@triton.jit(do_not_specialize=['T'])
def kernel(..., T, ...):
    # T 可以安全修改
    if condition:
        T = new_value
```

### 3. 保持计算一致性

确保优化后的内核保留：
- 相同的数学运算
- 相同的内存访问模式  
- 相同的控制流结构

### 4. 避免运行时条件判断

将编译时可确定的条件移动到内核参数中：

```python
# 作为constexpr参数传递
@triton.jit
def kernel(..., IS_VARLEN: tl.constexpr, ...):
    # 编译器可以根据IS_VARLEN值进行优化
```

## 准确性验证

### 参考实现模式

基于`ori1`主函数，创建准确性验证：

```python
def main():
    # 设置测试参数
    B, T, H, K, V = 2, 4096, 32, 256, 128
    q = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
    v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
    
    # 运行参考实现
    print("参考实现开始")
    output_ref, final_state_ref = original_function(q=q, k=k, v=v)
    
    # 运行优化实现
    print("优化实现开始")
    output_opt, final_state_opt = optimized_function(q=q, k=k, v=v)
    
    # 比较结果
    print(f"参考输出形状: {output_ref.shape}")
    print(f"优化输出形状: {output_opt.shape}")
    
    print(f"参考 min/max/mean: {output_ref.min():.6f}, {output_ref.max():.6f}, {output_ref.mean():.6f}")
    print(f"优化 min/max/mean: {output_opt.min():.6f}, {output_opt.max():.6f}, {output_opt.mean():.6f}")
    
    # 计算差异
    diff = torch.max(torch.abs(output_ref - output_opt))
    print(f'最大差异: {diff:.6e}')
    
    # 验证容差
    torch.testing.assert_close(output_ref, output_opt, atol=1e-3, rtol=1e-3)
    print("✓ 准确性验证通过")
```

## 常见陷阱和解决方案

### 1. 错误的任务映射
**问题**：步长计算错误导致索引计算不正确
**解决方案**：仔细检查网格维度和逻辑索引之间的关系

### 2. 性能下降
**问题**：任务分发开销降低性能
**解决方案**：确保任务粒度适当（不要太细）

### 3. 内存对齐问题
**问题**：NPU需要特定的内存对齐
**解决方案**：检查并确保张量维度满足NPU对齐要求

### 4. 逻辑运算符错误
**问题**：在掩码操作中使用`and`/`or`而不是`&`/`|`
**解决方案**：在Triton内核中将所有逻辑运算符替换为位运算符

### 5. 不必要的条件逻辑
**问题**：在NPU优化内核中引入额外的条件检查
**解决方案**：保持原始的参数修改模式，而不是创建新变量

## 模板文件

该技能包含参考模板：

1. **优化模板**：`optimize_template.py` - 展示完整的优化模式
2. **验证模板**：`validate_accuracy.py` - 展示准确性比较模式
3. **测试模板**：`test_complex_operator.py` - 复杂算子的综合测试

## 最佳实践

1. **从简单开始**：使用小问题规模进行验证
2. **增量测试**：独立测试每个优化步骤
3. **准确性优先**：在性能优化前确保数值正确性
4. **性能分析**：使用NPU分析工具识别瓶颈
5. **记录更改**：跟踪所有修改以便将来参考

## 示例

查看`triton_demo`目录获取完整示例：
- `ori1`：原始的fused_recurrent_fwd内核（GPU风格）
- `new1`：优化的fused_recurrent_fwd内核（NPU风格）

该优化展示了：
1. 从3D到1D物理核心网格的网格适配
2. NPU核心之间的任务分发
3. 容差检查的准确性验证
4. 通过更好的NPU利用率提高性能

## 优化案例总结

### 优化目标
将原始的GPU风格Triton算子 `chunk_fwd_o` 优化为NPU版本，同时避免引入额外的判断逻辑影响性能。

### 原始代码分析 (ori2.py)

#### 关键特性
1. **网格结构**: 3D逻辑网格 `(NV, NT, B * H)`
2. **参数处理**: 在`IS_VARLEN`分支中直接修改`T`参数
3. **装饰器**: 使用`@triton.jit(do_not_specialize=['T'])`允许参数修改

#### 性能关键点
```python
if IS_VARLEN:
    T = eos - bos  # 直接修改T参数，避免后续条件判断
```

### 优化过程

#### 第一阶段：网格适配优化
1. **网格转换**: 3D逻辑网格 → 1D物理核心网格
2. **任务分发**: 添加任务分发循环，均匀分配计算任务
3. **索引重建**: 从`task_id`正确重建原始索引`(i_v, i_t, i_bh)`

#### 第二阶段：性能优化修复
**问题发现**: 初始优化版本引入了`T_local`变量和多个条件判断：
```python
# 错误模式：增加了额外判断
T_local = eos - bos
p_q = tl.make_block_ptr(q_ptr, (T_local if IS_VARLEN else T, K), ...)
```

**修复方案**: 恢复原始的参数修改模式：
```python
# 正确模式：保持原始逻辑
T = eos - bos  # 直接修改T参数
p_q = tl.make_block_ptr(q_ptr, (T, K), ...)  # 直接使用T
```

### 优化后的代码 (new2.py)

#### 核心优化
1. **NPU网格适配**: `grid = (num_core,)`
2. **任务分发机制**: `for task_id in tl.range(core_id, task_num, num_core):`
3. **索引重建算法**: 正确计算`i_v`, `i_t`, `i_bh`
4. **性能保持**: 无额外条件判断，与原始代码逻辑一致

#### 关键改进
```python
# 保持原始的参数修改模式
if IS_VARLEN:
    T = eos - bos  # 直接修改T，与原始代码一致

# 后续代码直接使用T，无额外判断
p_q = tl.make_block_ptr(q_ptr, (T, K), ...)
m_t = o_t < T
```

### 性能优势

#### 保持的优势
1. ✅ **无额外判断逻辑**: 与原始代码相同的条件结构
2. ✅ **计算逻辑一致**: 相同的数学运算和内存访问
3. ✅ **参数处理一致**: 相同的参数修改模式

#### 新增的优势
1. ✅ **更好的NPU利用率**: 1D物理网格适配NPU架构
2. ✅ **任务均衡分发**: 计算任务均匀分配到核心
3. ✅ **减少启动开销**: 单次内核启动代替多次

### 使用建议

#### 部署使用
```python
# 使用优化版本
from new2 import chunk_fwd_o_npu as chunk_fwd_o

# API完全兼容
output = chunk_fwd_o(q, k, v, h, ...)
```

#### 性能调优
1. 根据实际NPU硬件调整`num_core`
2. 使用`validate_accuracy()`验证数值正确性
3. 参考`performance_optimization_guide.md`进行进一步优化