import pandas as pd
import pyarrow.parquet as pq
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def verify_parquet():
    # 使用pandas读取
    print("使用pandas读取parquet文件...")
    df = pd.read_parquet('health_bench/processed/train.parquet')
    print(f"\n数据集大小: {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查第一条数据的结构
    print("\n第一条数据的结构:")
    first_row = df.iloc[0]
    for col in df.columns:
        print(f"\n{col}:")
        value = first_row[col]
        if isinstance(value, (np.ndarray, pd.Series)):
            value = value.tolist()
        print(json.dumps(value, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    
    # 使用pyarrow读取
    print("\n使用pyarrow读取parquet文件...")
    table = pq.read_table('health_bench/processed/train.parquet')
    print(f"\nSchema:\n{table.schema}")
    
    # 验证所有必要字段
    required_fields = ["data_source", "prompt", "ability", "reward_model"]
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        print(f"\n警告：缺少必要字段: {missing_fields}")
    else:
        print("\n✓ 所有必要字段都存在")
    
    # 检查数据完整性
    print("\n数据完整性检查:")
    for field in required_fields:
        null_count = df[field].isnull().sum()
        print(f"{field}: {null_count} 个空值")

if __name__ == '__main__':
    verify_parquet() 