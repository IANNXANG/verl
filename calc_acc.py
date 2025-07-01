#!/usr/bin/env python3

import json
import sys

def calc_accuracy(jsonl_file):
    scores = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'score' in data:
                    scores.append(data['score'])
    
    if not scores:
        print("没有找到score数据")
        return
    
    accuracy = sum(scores) / len(scores)
    correct = sum(1 for s in scores if s > 0.5)
    total = len(scores)
    
    print(f"文件: {jsonl_file}")
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python calc_acc.py <jsonl_file>")
        sys.exit(1)
    
    calc_accuracy(sys.argv[1]) 