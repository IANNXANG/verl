#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 GSM8K 多轮工具数据集的脚本
按结构读取前5条数据
"""

import pandas as pd
import json
from pathlib import Path

def read_gsm8k_multiturn_data(data_path, num_samples=5):
    """
    读取 GSM8K 多轮工具数据集
    
    Args:
        data_path: 数据文件路径 (parquet 格式)
        num_samples: 要读取的样本数量
    
    Returns:
        读取的数据列表
    """
    try:
        # 读取 parquet 文件
        df = pd.read_parquet(data_path)
        
        print(f"数据集总共有 {len(df)} 条记录")
        print(f"读取前 {num_samples} 条数据:\n")
        
        # 获取前 num_samples 条数据
        sample_data = df.head(num_samples)
        
        # 按结构化方式显示每条数据
        for idx, row in sample_data.iterrows():
            print(f"=== 第 {idx + 1} 条数据 ===")
            
            # 基本信息
            print(f"数据源: {row.get('data_source', 'N/A')}")
            print(f"能力类型: {row.get('ability', 'N/A')}")
            
            # 对话内容
            print("\n对话内容:")
            if 'prompt' in row and isinstance(row['prompt'], list):
                for i, message in enumerate(row['prompt']):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    print(f"  {i+1}. [{role.upper()}]: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            # 奖励模型信息
            print("\n奖励模型:")
            if 'reward_model' in row:
                reward_info = row['reward_model']
                if isinstance(reward_info, dict):
                    print(f"  类型: {reward_info.get('style', 'N/A')}")
                    print(f"  标准答案: {reward_info.get('ground_truth', 'N/A')}")
            
            # 额外信息
            print("\n额外信息 (extra_info):")
            if 'extra_info' in row:
                extra_info = row['extra_info']
                if isinstance(extra_info, dict):
                    print(f"  数据分割: {extra_info.get('split', 'N/A')}")
                    print(f"  索引: {extra_info.get('index', 'N/A')}")
                    print(f"  原始问题: {extra_info.get('question', 'N/A')[:100]}{'...' if len(str(extra_info.get('question', ''))) > 100 else ''}")
                    print(f"  原始答案: {extra_info.get('answer', 'N/A')[:100]}{'...' if len(str(extra_info.get('answer', ''))) > 100 else ''}")
                    print(f"  需要工具参数: {extra_info.get('need_tools_kwargs', 'N/A')}")
                    
                    # 工具配置
                    if 'tools_kwargs' in extra_info:
                        print("  工具配置:")
                        tools_kwargs = extra_info['tools_kwargs']
                        if isinstance(tools_kwargs, dict):
                            for tool_name, tool_config in tools_kwargs.items():
                                print(f"    - {tool_name}:")
                                if isinstance(tool_config, dict) and 'create_kwargs' in tool_config:
                                    create_kwargs = tool_config['create_kwargs']
                                    print(f"      创建参数: {create_kwargs}")
            
            print("\n" + "="*50 + "\n")
        
        return sample_data.to_dict('records')
        
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        return None
    except Exception as e:
        print(f"读取数据时发生错误: {str(e)}")
        return None

def main():
    """
    主函数
    """
    # 数据文件路径 - 根据实际情况调整
    data_paths = [
        "/c22940/zy/code/verl/data/gsm8k_multiturn_w_tool/train.parquet",
        "./data/gsm8k_multiturn_w_tool/train.parquet",
        "~/data/gsm8k_multiturn_w_tool/train.parquet"
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        print("错误: 找不到 GSM8K 多轮工具数据文件")
        print("请确保数据文件存在于以下路径之一:")
        for path in data_paths:
            print(f"  - {path}")
        print("\n或者运行数据预处理脚本生成数据:")
        print("  python examples/data_preprocess/gsm8k_multiturn_w_tool.py")
        return
    
    print(f"使用数据文件: {data_path}\n")
    
    # 读取前5条数据
    data = read_gsm8k_multiturn_data(data_path, num_samples=5)
    
    if data:
        print(f"成功读取 {len(data)} 条数据")
    else:
        print("数据读取失败")

if __name__ == "__main__":
    main()