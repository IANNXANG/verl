import pandas as pd
import os

def read_parquet_files():
    """读取GSM8K数据集的train和test文件，统计数据条数并显示样例"""
    
    # 文件路径
    train_file = "/c22940/zy/code/verl/data/gsm8k/train.parquet"
    test_file = "/c22940/zy/code/verl/data/gsm8k/test.parquet"
    
    print("=" * 60)
    print("GSM8K 数据集分析")
    print("=" * 60)
    
    # 检查文件是否存在
    for file_path, file_type in [(train_file, "训练集"), (test_file, "测试集")]:
        if not os.path.exists(file_path):
            print(f"❌ {file_type}文件不存在: {file_path}")
            continue
            
        print(f"\n📁 读取{file_type}: {file_path}")
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(file_path)
            
            # 显示基本信息
            print(f"📊 数据条数: {len(df)}")
            print(f"📋 列名: {list(df.columns)}")
            print(f"💾 数据形状: {df.shape}")
            
            # 显示数据类型
            print(f"\n🔍 数据类型:")
            for col in df.columns:
                print(f"  - {col}: {df[col].dtype}")
            
            # 显示前3条数据样例
            print(f"\n📝 前3条数据样例:")
            for i in range(min(3, len(df))):
                print(f"\n--- 样例 {i+1} ---")
                for col in df.columns:
                    value = df.iloc[i][col]
                    # 如果文本太长，只显示前200个字符
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    print(f"{col}: {value}")
            
            # 显示统计信息
            if len(df) > 0:
                print(f"\n📈 统计信息:")
                for col in df.columns:
                    if df[col].dtype == 'object':  # 字符串列
                        avg_length = df[col].str.len().mean()
                        max_length = df[col].str.len().max()
                        min_length = df[col].str.len().min()
                        print(f"  - {col} 平均长度: {avg_length:.1f}, 最大长度: {max_length}, 最小长度: {min_length}")
                        
        except Exception as e:
            print(f"❌ 读取{file_type}时出错: {str(e)}")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    read_parquet_files()