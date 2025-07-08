import pandas as pd
import numpy as np
from healthbench_reward_fn import compute_score

def convert_numpy_to_python(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return convert_numpy_to_python(obj.to_dict())
    else:
        return obj

def test_reward_function():
    # 读取parquet数据
    df = pd.read_parquet('health_bench/processed/train.parquet')
    first_row = df.iloc[0]
    
    # 构造测试数据，转换为Python原生类型
    test_data = {
        "data_source": first_row['data_source'],
        "prompt": convert_numpy_to_python(first_row['prompt']),
        "reward_model": convert_numpy_to_python(first_row['reward_model'])
    }
    
    # 测试用例1：好的回答
    good_response = """
    Cutaneous leishmaniasis is indeed a likely diagnosis for multiple non-healing ulcers in a patient from a desert region in North Africa. This parasitic infection is transmitted by sandfly bites and is common in arid regions.

    The disease typically presents as papules that evolve into painless ulcers with raised edges and a central crater. For proper diagnosis, we would need a skin biopsy or lesion smear to detect amastigotes, or PCR testing.

    Treatment options include:
    1. Local treatment (for uncomplicated cases):
       - Cryotherapy
       - Topical paromomycin
       - Heat therapy

    2. Systemic treatment (for multiple/complicated lesions):
       - Pentavalent antimonials (first-line)
       - Liposomal amphotericin B
       - Miltefosine
       - Azole antifungals

    The choice of treatment depends on factors like lesion size, number, location, and immune status. Regular wound care and debridement are also important.
    """
    
    # 测试用例2：较差的回答
    poor_response = "Take some antibiotics and it will be fine."
    
    # 计算分数
    good_score = compute_score(
        data_source=test_data["data_source"],
        solution_str=good_response,
        extra_info=test_data
    )
    
    poor_score = compute_score(
        data_source=test_data["data_source"],
        solution_str=poor_response,
        extra_info=test_data
    )
    
    print(f"好的回答得分: {good_score:.2f}")
    print(f"较差的回答得分: {poor_score:.2f}")
    
    # 打印测试数据结构
    print("\n测试数据结构:")
    for key, value in test_data.items():
        print(f"\n{key}:")
        if isinstance(value, (list, dict)):
            print(f"{type(value)}: {value}")
        else:
            print(f"{type(value)}: {value}")

if __name__ == '__main__':
    test_reward_function() 