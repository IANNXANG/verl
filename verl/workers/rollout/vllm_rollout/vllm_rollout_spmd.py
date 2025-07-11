# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: List[str]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d.get("tags", [])
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

def _generate_rubric_system_message(rubric_items: List[RubricItem], rubric_ratio: float) -> str:
    """Generate system message with rubric information for open-book evaluation."""
    if not rubric_items:
        return ""
    
    # 如果rubric_ratio < 1.0，随机选择部分criterion
    if rubric_ratio < 1.0:
        total_criteria = len(rubric_items)
        num_to_show = round(total_criteria * rubric_ratio)  # 四舍五入
        if num_to_show == 0 and total_criteria > 0 and rubric_ratio > 0:
            num_to_show = 1  # 至少显示一个criterion（但ratio=0时除外）
        
        # 随机选择要显示的criterion
        if num_to_show > 0:
            selected_rubric_items = random.sample(rubric_items, num_to_show)
        else:
            selected_rubric_items = []
    else:
        selected_rubric_items = rubric_items
    
    positive_points = []
    negative_points = []
    
    pos_count = 1
    neg_count = 1
    
    for rubric_item in selected_rubric_items:
        criterion = rubric_item.criterion
        points = rubric_item.points
        
        if points > 0:
            positive_points.append(f"Criterion {pos_count}: {criterion} (worth {points} points)")
            pos_count += 1
        elif points < 0:
            negative_points.append(f"Criterion {neg_count}: {criterion} (penalty: {abs(points)} points)")
            neg_count += 1
    
    rubric_message = "You are a helpful medical assistant. For this question, please consider the following evaluation criteria:\n\n"
    
    if positive_points:
        rubric_message += "IMPORTANT POINTS TO INCLUDE (you should aim to address these):\n"
        rubric_message += "\n".join(positive_points)
        rubric_message += "\n\n"
    
    if negative_points:
        rubric_message += "IMPORTANT POINTS TO AVOID (you should not do these):\n"
        rubric_message += "\n".join(negative_points)
        rubric_message += "\n\n"
    
    rubric_message += "Please provide a comprehensive and helpful response that addresses the patient's concerns while following the above guidelines.\n\n"
    
    rubric_message += "IMPORTANT: Do not mention or reference these evaluation criteria in your response. Do not indicate that you have seen any scoring rubric or evaluation guidelines. Your response should appear natural and spontaneous. Revealing that you have access to evaluation criteria would be considered cheating and is strictly prohibited."
    
    return rubric_message

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # 分层系统提示词功能
        enable_graded_system_prompt = getattr(self.config, 'enable_graded_system_prompt', False)
        original_n = self.sampling_params.n
        original_vllm_inputs = vllm_inputs.copy()
        original_lora_requests = lora_requests.copy() if lora_requests else None
        
        if enable_graded_system_prompt and do_sample and not is_validate and original_n > 1:
            print(f"[GRADED SYSTEM PROMPT] 启用分层系统提示词功能，n={original_n}")
            
            # 从meta_info中获取rubric信息（额外数据通道）
            rubric_info_available = False
            rubric_items_list = []
            
            # 优先从meta_info获取reward_model信息
            if 'graded_system_prompt_reward_models' in prompts.meta_info:
                print(f"[GRADED SYSTEM PROMPT] 成功从meta_info获取rubric数据")
                reward_models = prompts.meta_info['graded_system_prompt_reward_models']
                
                for i in range(batch_size):
                    if i < len(reward_models):
                        reward_model = reward_models[i]
                        if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                            rubrics = reward_model['rubrics']
                            rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                            rubric_items_list.append(rubric_items)
                            rubric_info_available = True
                        else:
                            rubric_items_list.append([])
                    else:
                        rubric_items_list.append([])
            
            # 备用方案：从non_tensor_batch中获取rubric信息
            elif 'reward_model' in non_tensor_batch:
                print(f"[GRADED SYSTEM PROMPT] 从non_tensor_batch获取rubric数据（备用通道）")
                
                for i in range(batch_size):
                    reward_model = non_tensor_batch['reward_model'][i]
                    if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                        rubrics = reward_model['rubrics']
                        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                        rubric_items_list.append(rubric_items)
                        rubric_info_available = True
                    else:
                        rubric_items_list.append([])
            else:
                print(f"[GRADED SYSTEM PROMPT] 未找到rubric数据")
            
            # 如果non_tensor_batch中没有，尝试从其他地方获取
            if not rubric_info_available and hasattr(prompts, 'non_tensor_batch'):
                # 检查extra_info
                if 'extra_info' in prompts.non_tensor_batch:
                    for i in range(batch_size):
                        extra_info = prompts.non_tensor_batch['extra_info'][i]
                        if isinstance(extra_info, dict) and 'reward_model' in extra_info:
                            reward_model = extra_info['reward_model']
                            if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                                rubrics = reward_model['rubrics']
                                rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                                rubric_items_list.append(rubric_items)
                                rubric_info_available = True
                            else:
                                rubric_items_list.append([])
                        else:
                            rubric_items_list.append([])
            
            if not rubric_info_available:
                print(f"[GRADED SYSTEM PROMPT] 警告：未找到rubric信息，将使用默认行为")
                # 如果没有rubric信息，填充空列表
                rubric_items_list = [[] for _ in range(batch_size)]
            else:
                total_rubrics = sum(len(items) for items in rubric_items_list)
                print(f"[GRADED SYSTEM PROMPT] 成功提取{total_rubrics}个rubric items")
            
            # 生成n个不同比例的system prompt
            expanded_vllm_inputs = []
            expanded_lora_requests = []
            
            for i in range(batch_size):
                rubric_items = rubric_items_list[i]
                original_prompt_ids = original_vllm_inputs[i]["prompt_token_ids"]
                
                for sample_idx in range(original_n):
                    # 计算rubric透露比例：第一次是1，第二次是(n-2)/(n-1)，...，最后一次是0
                    if original_n == 1:
                        rubric_ratio = 1.0
                    else:
                        rubric_ratio = max(0.0, (original_n - 1 - sample_idx) / (original_n - 1))
                    
                    # 生成system prompt
                    if rubric_items and rubric_ratio > 0:
                        system_message = _generate_rubric_system_message(rubric_items, rubric_ratio)
                        print(f"[GRADED SYSTEM PROMPT] 样本{i}-{sample_idx}: rubric_ratio={rubric_ratio:.2f}, 系统提示词长度={len(system_message)}")
                        print(f"[GRADED SYSTEM PROMPT] 系统提示词内容:\n{system_message[:200]}..." if len(system_message) > 200 else f"[GRADED SYSTEM PROMPT] 系统提示词内容:\n{system_message}")
                        
                        # 将system message转换为token ids并添加到prompt前面
                        # 使用tokenizer来编码system message
                        tokenizer = self.inference_engine.llm_engine.tokenizer
                        system_tokens = tokenizer.encode(system_message, add_special_tokens=False)
                        
                        # 组合system tokens和原始prompt tokens
                        combined_prompt_ids = system_tokens + original_prompt_ids
                        print(f"[GRADED SYSTEM PROMPT] 原始prompt长度: {len(original_prompt_ids)}, 系统提示词token长度: {len(system_tokens)}, 组合后长度: {len(combined_prompt_ids)}")
                    else:
                        combined_prompt_ids = original_prompt_ids
                        print(f"[GRADED SYSTEM PROMPT] 样本{i}-{sample_idx}: 无系统提示词 (rubric_ratio={rubric_ratio:.2f})")
                    
                    # 创建新的vllm input
                    new_input = original_vllm_inputs[i].copy()
                    new_input["prompt_token_ids"] = combined_prompt_ids
                    expanded_vllm_inputs.append(new_input)
                    
                    # 添加对应的lora request
                    if original_lora_requests:
                        expanded_lora_requests.append(original_lora_requests[i])
            
            # 更新参数
            vllm_inputs = expanded_vllm_inputs
            lora_requests = expanded_lora_requests if expanded_lora_requests else None
            batch_size = len(vllm_inputs)  # 新的batch size = 原batch_size * n
            
            # 修改sampling参数，每个prompt只采样1次
            kwargs_with_n1 = kwargs.copy()
            kwargs_with_n1["n"] = 1
            
            print(f"[GRADED SYSTEM PROMPT] 扩展后batch_size: {batch_size}, 每个prompt采样1次")
        else:
            kwargs_with_n1 = kwargs

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs_with_n1):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)
                # 处理reward_model字段
                if "reward_model" in non_tensor_batch.keys():
                    non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], self.sampling_params.n)

            # 如果使用了分层系统提示词，需要恢复原始的prompt信息
            if enable_graded_system_prompt and do_sample and not is_validate and original_n > 1:
                print(f"[GRADED SYSTEM PROMPT] 恢复原始prompt信息，当前batch_size: {batch_size}")
                
                # 重新构建原始的idx, attention_mask, position_ids
                original_batch_size = len(original_vllm_inputs)
                
                # 恢复原始的prompt token ids
                restored_prompts = []
                for i in range(original_batch_size):
                    original_prompt_ids = original_vllm_inputs[i]["prompt_token_ids"]
                    # 重复n次
                    for _ in range(original_n):
                        restored_prompts.append(original_prompt_ids)
                
                # 注意：这里需要使用与当前idx相同的max_len来确保尺寸一致
                current_max_len = idx.shape[1]  # 使用当前batch中已有的最大长度
                restored_idx = []
                restored_attention_mask = []
                
                for prompt_ids in restored_prompts:
                    # 左padding到current_max_len长度
                    if len(prompt_ids) > current_max_len:
                        # 如果原始prompt比当前长度还长，进行截断（左截断保留最后的tokens）
                        padded_ids = prompt_ids[-current_max_len:]
                        attention = [1] * current_max_len
                    else:
                        # 左padding
                        padding_length = current_max_len - len(prompt_ids)
                        padded_ids = [self.pad_token_id] * padding_length + prompt_ids
                        attention = [0] * padding_length + [1] * len(prompt_ids)
                    
                    restored_idx.append(padded_ids)
                    restored_attention_mask.append(attention)
                
                restored_idx = torch.tensor(restored_idx, device=idx.device)
                restored_attention_mask = torch.tensor(restored_attention_mask, device=attention_mask.device)
                
                # 重新计算position_ids
                restored_position_ids = (restored_attention_mask.cumsum(dim=1) - 1) * restored_attention_mask
                
                # 更新idx, attention_mask, position_ids为原始值
                idx = restored_idx
                attention_mask = restored_attention_mask  # 只是prompt部分的attention_mask
                position_ids = restored_position_ids
                
                print(f"[GRADED SYSTEM PROMPT] 恢复完成，idx shape: {idx.shape}, attention_mask shape: {attention_mask.shape}")
                
                # 同时需要重复non_tensor_batch中的数据
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], original_n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], original_n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], original_n)
                # 处理reward_model字段 - 这是修复关键！
                if "reward_model" in non_tensor_batch.keys():
                    non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], original_n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
