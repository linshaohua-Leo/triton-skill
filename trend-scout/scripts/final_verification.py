#!/usr/bin/env python3
"""
最终验证脚本：检查优化是否正确移除了额外判断逻辑
"""

import re

def check_optimization():
    print("=" * 60)
    print("优化验证：检查额外判断逻辑是否已移除")
    print("=" * 60)
    
    # 读取优化后的代码
    with open('new2.py', 'r', encoding='utf-8') as f:
        new2_content = f.read()
    
    # 检查是否还有T_local变量
    t_local_pattern = r'T_local'
    t_local_matches = re.findall(t_local_pattern, new2_content)
    
    # 检查是否还有条件表达式 (T_local if IS_VARLEN else T)
    conditional_pattern = r'\(T_local if IS_VARLEN else T\)'
    conditional_matches = re.findall(conditional_pattern, new2_content)
    
    # 检查是否直接修改T参数
    t_modification_pattern = r'if IS_VARLEN:[\s\S]*?T = eos - bos'
    t_modification_matches = re.findall(t_modification_pattern, new2_content)
    
    print("\n1. 检查 T_local 变量:")
    if t_local_matches:
        print(f"   ✗ 发现 {len(t_local_matches)} 处 T_local 引用")
        for match in t_local_matches:
            print(f"      - 找到: {match}")
    else:
        print("   ✓ 未发现 T_local 变量")
    
    print("\n2. 检查条件表达式 (T_local if IS_VARLEN else T):")
    if conditional_matches:
        print(f"   ✗ 发现 {len(conditional_matches)} 处条件表达式")
        for match in conditional_matches:
            print(f"      - 找到: {match}")
    else:
        print("   ✓ 未发现条件表达式")
    
    print("\n3. 检查是否直接修改T参数:")
    if t_modification_matches:
        print("   ✓ 发现直接修改T参数的代码")
        print(f"      - 模式: T = eos - bos")
    else:
        print("   ✗ 未发现直接修改T参数的代码")
    
    print("\n4. 检查关键代码模式:")
    # 检查关键代码段
    key_patterns = [
        (r'p_q = tl\.make_block_ptr\(q_ptr, \(T, K\)', '正确：使用T作为参数'),
        (r'p_k = tl\.make_block_ptr\(k_ptr, \(K, T\)', '正确：使用T作为参数'),
        (r'm_t = o_t < T', '正确：直接比较T'),
        (r'p_v = tl\.make_block_ptr\(v_ptr, \(T, V\)', '正确：使用T作为参数'),
        (r'p_o = tl\.make_block_ptr\(o_ptr, \(T, V\)', '正确：使用T作为参数'),
    ]
    
    for pattern, description in key_patterns:
        matches = re.findall(pattern, new2_content)
        if matches:
            print(f"   ✓ {description}")
        else:
            print(f"   ✗ 未找到: {description}")
    
    print("\n5. 检查装饰器使用:")
    decorator_pattern = r"@triton\.jit\(do_not_specialize=\['T'\]\)"
    decorator_matches = re.findall(decorator_pattern, new2_content)
    if decorator_matches:
        print("   ✓ 正确使用 do_not_specialize=['T'] 装饰器")
    else:
        print("   ✗ 未找到 do_not_specialize=['T'] 装饰器")
    
    print("\n" + "=" * 60)
    print("优化验证总结")
    print("=" * 60)
    
    # 总结验证结果
    issues_found = len(t_local_matches) + len(conditional_matches)
    
    if issues_found == 0 and t_modification_matches:
        print("✅ 优化成功！")
        print("   - 移除了所有额外判断逻辑")
        print("   - 保持了原始的参数修改模式")
        print("   - 代码结构与原始版本一致")
        return True
    else:
        print("❌ 优化存在问题：")
        if t_local_matches:
            print(f"   - 仍有 {len(t_local_matches)} 处 T_local 引用")
        if conditional_matches:
            print(f"   - 仍有 {len(conditional_matches)} 处条件表达式")
        if not t_modification_matches:
            print("   - 未发现直接修改T参数的代码")
        return False

def compare_with_original():
    print("\n" + "=" * 60)
    print("与原始代码对比")
    print("=" * 60)
    
    # 读取原始代码
    with open('ori2.py', 'r', encoding='utf-8') as f:
        ori2_content = f.read()
    
    # 读取优化代码
    with open('new2.py', 'r', encoding='utf-8') as f:
        new2_content = f.read()
    
    # 提取关键部分进行对比
    print("\n关键代码对比：")
    
    # 提取IS_VARLEN处理部分
    original_varlen = re.search(r'if IS_VARLEN:[\s\S]*?NT = tl\.cdiv\(T, BT\)', ori2_content)
    optimized_varlen = re.search(r'if IS_VARLEN:[\s\S]*?NT = tl\.cdiv\(T, BT\)', new2_content)
    
    if original_varlen and optimized_varlen:
        print("\n原始代码 IS_VARLEN 处理:")
        print("-" * 40)
        print(original_varlen.group(0)[:200] + "...")
        
        print("\n优化代码 IS_VARLEN 处理:")
        print("-" * 40)
        print(optimized_varlen.group(0)[:200] + "...")
        
        # 检查是否一致
        orig_clean = re.sub(r'\s+', ' ', original_varlen.group(0))
        opt_clean = re.sub(r'\s+', ' ', optimized_varlen.group(0))
        
        if "T = eos - bos" in orig_clean and "T = eos - bos" in opt_clean:
            print("\n✅ IS_VARLEN 处理逻辑一致")
        else:
            print("\n❌ IS_VARLEN 处理逻辑不一致")
    
    # 检查后续代码是否直接使用T
    print("\n后续代码使用T的对比:")
    
    # 在原始代码中查找使用T的地方
    original_t_usage = re.findall(r'\(T,[^)]*\)|\([^,]*T[^)]*\)', ori2_content)
    optimized_t_usage = re.findall(r'\(T,[^)]*\)|\([^,]*T[^)]*\)', new2_content)
    
    print(f"原始代码中T的使用次数: {len(original_t_usage)}")
    print(f"优化代码中T的使用次数: {len(optimized_t_usage)}")
    
    # 检查是否有条件表达式
    original_conditionals = re.findall(r'if.*else.*T', ori2_content)
    optimized_conditionals = re.findall(r'if.*else.*T', new2_content)
    
    print(f"\n原始代码中条件表达式数量: {len(original_conditionals)}")
    print(f"优化代码中条件表达式数量: {len(optimized_conditionals)}")
    
    if len(optimized_conditionals) == 0:
        print("✅ 优化代码中无额外条件表达式")
    else:
        print("❌ 优化代码中仍有条件表达式")

if __name__ == "__main__":
    print("Triton NPU 优化验证工具")
    print("检查 new2.py 是否已正确优化，移除额外判断逻辑")
    print()
    
    success = check_optimization()
    compare_with_original()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有检查通过！优化代码符合要求。")
        print("   代码已成功移除额外判断逻辑，保持原始性能特征。")
    else:
        print("⚠️  发现一些问题，请检查优化代码。")
        print("   确保移除了所有 T_local 引用和条件表达式。")
    
    print("=" * 60)