import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from healthbench_reward import RubricItem

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def make_map_fn(split: str):
    """构造数据映射函数"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # 提取prompt
        prompt = example['prompt']
        
        # 提取rubrics
        rubrics = [RubricItem.from_dict(r) for r in example['rubrics']]
        
        # 构造reward_model字段
        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubrics],
            "ground_truth": ""  # 使用空字符串
        }
        
        # 构造verl所需的数据格式
        data = {
            "data_source": "healthbench",
            "prompt": prompt,  # 保留外层prompt
            "ability": "medical_chat",
            "reward_model": reward_model,  # 保留外层reward_model
            "extra_info": {
                "prompt": prompt,  # 在extra_info中也放入prompt
                "reward_model": reward_model  # 在extra_info中也放入reward_model
            }
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    """处理数据集"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # 移除所有原始列
    )
    return processed_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='health_bench/data_group')
    parser.add_argument('--output_dir', default='data/health_bench')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # 加载训练数据
    train_data = []
    for i in range(1, 4):  # group1-3
        file_path = os.path.join(args.local_dir, f'healthbench_group{i}.jsonl')
        train_data.extend(load_jsonl(file_path))
    
    # 加载验证数据
    val_file = os.path.join(args.local_dir, 'healthbench_null.jsonl')
    val_data = load_jsonl(val_file)
    # 只保留前100笔验证数据“验证集”
    val_data = val_data[:100]
    
    # 处理训练集和验证集
    train_dataset = process_dataset(train_data, 'train')
    val_dataset = process_dataset(val_data, 'val')
    
    # 保存为parquet格式
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_train.parquet'))
    val_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_val.parquet'))
    
    # 打印数据集信息
    print("\n数据集信息:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 打印数据样本示例
    print("\n训练集样本示例:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
    print("\n验证集样本示例:")
    print(json.dumps(val_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main() 