import json
from typing import Dict, Any, List
from healthbench_reward import (
    RubricItem,
    grade_single_example,
    ChatCompletionSampler,
    VLLMSampler
)

def init_grader():
    """初始化评分模型"""
    # 这里使用VLLM作为评分模型
    return VLLMSampler(
        max_tokens=2048,
        enable_thinking=False,
        filter_think_tags=True
    )

# 全局评分模型实例
GRADER = init_grader()

def compute_score(data_source: str, solution_str: str, ground_truth: str = None, extra_info: Dict[str, Any] = None) -> float:
    """
    计算healthbench奖励分数
    
    Args:
        data_source: 数据集名称（从DataProto.non_tensor_batch['data_source']获取）
        solution_str: 模型生成的回答
        ground_truth: 不使用
        extra_info: 不使用，所有信息都从DataProto中获取
    
    Returns:
        float: 奖励分数 [0, 1]
    """
    if data_source != "healthbench":
        return 0.0
    
    # 从DataProto.non_tensor_batch中获取数据
    # 在verl中，这些信息会自动从parquet文件加载并传入
    try:
        # 获取prompt和reward_model
        prompt = extra_info.get("prompt", [])
        reward_model = extra_info.get("reward_model", {})
        rubrics = reward_model.get("rubrics", [])
        
        if not prompt or not rubrics:
            return 0.0
            
        # 重建rubrics
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        # 使用grade_single_example计算分数
        score, _, _ = grade_single_example(
            prompt=prompt,
            response=solution_str,
            rubric_items=rubric_items,
            grader_model=GRADER
        )
        
        return score  # 已经是归一化的分数[0,1]
        
    except Exception as e:
        print(f"计算奖励分数时出错: {e}")
        return 0.0 