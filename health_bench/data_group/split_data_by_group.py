#!/usr/bin/env python3
"""
将HealthBench数据按照医生理想回答的分组分成四个文件
- none: 没有医生理想回答的样本
- Group 1: 无参考回答的医生理想回答
- Group 2: 有2024年8-9月模型参考回答的医生理想回答
- Group 3: 有2025年4月模型参考回答的医生理想回答
"""

import json
import os
from pathlib import Path

def split_healthbench_by_group(input_file: str, output_dir: str = "data_group"):
    """
    将HealthBench数据按照医生理想回答分组分成四个文件
    
    Args:
        input_file: 输入的jsonl文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 初始化分组数据
    groups = {
        'none': [],
        'Group 1': [],
        'Group 2': [], 
        'Group 3': []
    }
    
    # 读取数据并分组
    total_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total_count += 1
            
            if data['ideal_completions_data'] is None:
                groups['none'].append(data)
            else:
                group = data['ideal_completions_data']['ideal_completions_group']
                if group in groups:
                    groups[group].append(data)
                else:
                    print(f"Warning: Unknown group '{group}' found in data")
    
    # 写入分组文件
    file_mapping = {
        'none': 'healthbench_null.jsonl',
        'Group 1': 'healthbench_group1.jsonl',
        'Group 2': 'healthbench_group2.jsonl',
        'Group 3': 'healthbench_group3.jsonl'
    }
    
    for group, filename in file_mapping.items():
        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in groups[group]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"已保存 {len(groups[group])} 个样本到 {output_file}")
    
    print(f"\n总计处理 {total_count} 个样本")
    print("分组统计:")
    for group, data in groups.items():
        percentage = len(data) / total_count * 100
        print(f"  {group}: {len(data)} 个样本 ({percentage:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将HealthBench数据按医生理想回答分组")
    parser.add_argument("--input", "-i", 
                       default="2025-05-07-06-14-12_oss_eval.jsonl",
                       help="输入文件路径")
    parser.add_argument("--output", "-o", 
                       default="data_group",
                       help="输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        exit(1)
    
    print(f"正在处理文件: {args.input}")
    print(f"输出目录: {args.output}")
    print("-" * 50)
    
    split_healthbench_by_group(args.input, args.output)
    
    print("-" * 50)
    print("数据分组完成!") 