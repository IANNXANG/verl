# 分层系统提示词功能（Graded System Prompt）

## 功能概述

该功能为VERL框架增加了**分层系统提示词**能力，允许在GRPO训练过程中对同一个prompt的不同采样分配不同比例的规则透露到system prompt中，从而产生质量差异更大的回答，提高训练效果。

## 核心思想

在GRPO训练中，原本的采样方式是：
- 一个prompt，采样n次（如n=8），每次采样都使用相同的prompt

新的采样方式：
- 一个prompt，生成n个不同的system prompt（每个包含不同比例的评分规则）
- 每个带有不同system prompt的prompt只采样1次
- 最终返回的prompt信息是不带system prompt的原始prompt（不参与训练）

## 规则透露比例计算

对于n次采样：
- 第1次：比例 = 1.0（透露全部规则）
- 第2次：比例 = (n-2)/(n-1)
- 第3次：比例 = (n-3)/(n-1)
- ...
- 第n次：比例 = 0.0（不透露任何规则）

例如，当n=8时：
- 采样1：比例=1.0（透露100%规则）
- 采样2：比例=0.86（透露约86%规则）
- 采样3：比例=0.71（透露约71%规则）
- 采样4：比例=0.57（透露约57%规则）
- 采样5：比例=0.43（透露约43%规则）
- 采样6：比例=0.29（透露约29%规则）
- 采样7：比例=0.14（透露约14%规则）
- 采样8：比例=0.0（不透露规则）

## 使用方法

### 1. 配置参数

在配置文件中添加参数来启用功能：

```bash
actor_rollout_ref.rollout.enable_graded_system_prompt=True
```

### 2. 数据格式要求

数据需要包含rubric信息，格式如下：

```python
{
    "prompt": [{"role": "user", "content": "用户问题"}],
    "reward_model": {
        "rubrics": [
            {
                "criterion": "正确识别症状",
                "points": 8,
                "tags": ["clinical_assessment"]
            },
            {
                "criterion": "提供自我护理建议", 
                "points": 6,
                "tags": ["self_care"]
            },
            {
                "criterion": "建议处方药物",
                "points": -2,
                "tags": ["inappropriate"]
            }
        ]
    }
}
```

### 3. 生效条件

该功能只在以下条件同时满足时生效：
- `enable_graded_system_prompt=True`
- `do_sample=True`（进行采样）
- `is_validate=False`（非验证阶段）
- `n > 1`（采样次数大于1）

## 实现细节

### 关键文件修改

1. **配置文件** (`verl/trainer/config/ppo_trainer.yaml`)
   - 添加`enable_graded_system_prompt`参数

2. **Rollout模块** (`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`)
   - 添加`RubricItem`数据类
   - 添加`_generate_rubric_system_message`函数
   - 修改`generate_sequences`方法支持分层系统提示词

### System Prompt生成逻辑

```python
def _generate_rubric_system_message(rubric_items: List[RubricItem], rubric_ratio: float) -> str:
    # 根据比例随机选择要显示的criterion
    # 生成包含正面和负面评分标准的system prompt
    # 添加隐藏指令，要求模型不要在回答中提及评分标准
```

### 数据传递路径

1. 数据从parquet文件加载，包含rubric信息
2. 通过DataProto的non_tensor_batch传递到rollout模块
3. Rollout模块提取rubric信息，生成不同的system prompt
4. 将system prompt编码为token并与原始prompt拼接
5. 生成回答后，恢复原始prompt信息用于后续训练

## 预期效果

- **更大的回答差异**：有规则指导的回答质量更高，无规则指导的回答质量相对较低
- **更好的训练信号**：GRPO算法能够更好地学习到好坏回答之间的差异
- **保持训练一致性**：system prompt只在rollout阶段生效，不影响训练过程

## 注意事项

1. **性能影响**：功能启用后会增加推理时的计算量（需要编码system prompt）
2. **内存使用**：扩展prompt长度可能增加内存使用
3. **兼容性**：只支持包含rubric信息的数据集
4. **调试信息**：启用后会打印详细的调试信息，方便排查问题

## 示例输出

启用功能后，在日志中会看到类似输出：

```
[GRADED SYSTEM PROMPT] 启用分层系统提示词功能，n=8
[GRADED SYSTEM PROMPT] 从non_tensor_batch中找到reward_model字段
[GRADED SYSTEM PROMPT] 样本0-0: rubric_ratio=1.00, 系统提示词长度=1245
[GRADED SYSTEM PROMPT] 系统提示词内容:
You are a helpful medical assistant. For this question, please consider the following evaluation criteria:

IMPORTANT POINTS TO INCLUDE (you should aim to address these):
Criterion 1: 正确识别患者症状并给出合理的初步判断 (worth 8 points)
...
[GRADED SYSTEM PROMPT] 原始prompt长度: 156, 系统提示词token长度: 312, 组合后长度: 468
[GRADED SYSTEM PROMPT] 样本0-7: 无系统提示词 (rubric_ratio=0.00)
[GRADED SYSTEM PROMPT] 扩展后batch_size: 32, 每个prompt采样1次
[GRADED SYSTEM PROMPT] 恢复原始prompt信息，当前batch_size: 32
[GRADED SYSTEM PROMPT] 恢复完成，idx shape: torch.Size([32, 156]), attention_mask shape: torch.Size([32, 156])
``` 