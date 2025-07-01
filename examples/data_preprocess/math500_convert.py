#!/usr/bin/env python3
"""
Convert MATH-500 dataset to VERL format
"""

import argparse
import os
import datasets
import pandas as pd
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    """Extract solution from MATH format"""
    return remove_boxed(last_boxed_only_string(solution_str))


def convert_math500_to_verl_format(output_dir="~/data/math500"):
    """Convert MATH-500 dataset to VERL format"""
    
    # 创建输出目录
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在下载MATH-500数据集...")
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    data_source = "DigitalLearningGmbH/MATH-lighteval"  # 使用VERL能识别的MATH数据源
    
    print("正在转换数据格式...")
    
    converted_data = []
    for idx, example in enumerate(dataset):
        question = example["problem"]
        full_question = question + " " + instruction_following
        
        answer = example["solution"] 
        solution = extract_solution(answer)
        
        data_item = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": full_question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "original_dataset": "HuggingFaceH4/MATH-500",  # 仅保留标识信息
                "dataset_size": 500
            }
        }
        converted_data.append(data_item)
    
    # 保存转换后的数据
    output_file = os.path.join(output_dir, "test.parquet")
    df = pd.DataFrame(converted_data)
    df.to_parquet(output_file, index=False)
    
    print(f"✅ MATH-500数据集转换完成!")
    print(f"📁 保存位置: {output_file}")
    print(f"📊 数据量: {len(converted_data)} 个问题")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATH-500 to VERL format")
    parser.add_argument("--output_dir", default="~/data/math500", 
                       help="Output directory for converted dataset")
    
    args = parser.parse_args()
    convert_math500_to_verl_format(args.output_dir) 