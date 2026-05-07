import math
from typing import Optional, Tuple, Union, List, Dict, Sequence, Any
import random
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.cache_utils import Cache
import json
import copy
import numpy as np

from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.utils import ModelOutput, is_torchdynamo_compiling
from transformers.generation.utils import (
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessorList

from .logit_processor_3dim import MultiTokensVLLogitsProcessor, MultiTokensInterleavedTopKLogitsWarper, MultiTreeTokensVLLogitsProcessor, \
    get_double_cfg_input_ids, gather_from_split_tensors

from absl import logging
import time

# 无大改的函数
from scheduler.jacobi_iteration_lumina_mgpt import sampling_logits2tokens, check_is_force_no_cfg, Result
from scheduler.jacobi_iteration_lumina_mgpt import set_seed, postprocess_cfg_decode, get_draft_token_for_tree, TOPK

# 类别
from scheduler.jacobi_iteration_lumina_mgpt import SpeculativeSampler, SpeculativeGroupSumSampler, SpeculativeSamplerLantern, GroupSpeculativeSampler, GroupSpeculativeGroupSumSampler

from scheduler.jacobi_iteration_lumina_mgpt import prefix_matching_next_tokens, push_forward_model_kwargs_and_inputs
from scheduler.jacobi_iteration_lumina_mgpt import get_multi_token_for_preparation, renew_backbone

def renew_pipeline(model_class):
    class JacobiPipeline(model_class):

        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

        def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10, static_tree = False):
            cfg = self.cfg if hasattr(self, 'cfg') else cfg
            image_top_k = self.image_top_k if hasattr(self, 'image_top_k') else image_top_k
            text_top_k = self.text_top_k if hasattr(self, 'text_top_k') else text_top_k

            logits_processor = LogitsProcessorList()

            if static_tree:
                candidate_processor = MultiTreeTokensVLLogitsProcessor(
                    image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                    image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
                    image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
                    patch_size=32,
                    voc_size=self.model.config.vocab_size,
                    device = self.device,
                )
            else:
                candidate_processor = MultiTokensVLLogitsProcessor(
                    image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                    image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
                    image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
                    patch_size=32,
                    voc_size=self.model.config.vocab_size,
                    device = self.device,
                )

            topk_processor = MultiTokensInterleavedTopKLogitsWarper(
                image_top_k=image_top_k,
                text_top_k=text_top_k,
                image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
                image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            )

            logits_processor.append(candidate_processor)
            logits_processor.append(topk_processor)

            return logits_processor
    
    return JacobiPipeline

def renew_sampler(model_class):
    def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
        """
        Pad the given path list with a specific value up to a specified length.

        Parameters:
        - path (list): The original list that needs padding.
        - length (int): The desired length of the padded list.
        - pad_value (optional, default=-2): The value to use for padding.

        Returns:
        - list: A new list based on the original path but padded to the desired length.

        Example:
        >>> pad_path([1,2,3], 5)
        [1, 2, 3, -2, -2]

        Note:
        If the given path is already longer than the specified length,
        then no padding occurs, and the original path is returned.
        """

        # Calculate the number of padding values needed by subtracting the length
        # of the path from the desired length.
        # Append the padding values to the original path and return the new list.
        return path + [pad_value] * (length - len(path))
    def generate_tree_buffers(tree_choices, device="cuda"):
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                    dim=1)

        maxitem = retrieve_indices.max().item() + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        p_indices = torch.tensor(p_indices)
        p_indices_new = p_indices[retrieve_indices]
        p_indices_new = p_indices_new.tolist()

        b_indices = [[]] + b_indices
        b_indices_new = []
        for ib in range(retrieve_indices.shape[0]):
            iblist = []
            for jb in range(retrieve_indices.shape[1]):
                index = retrieve_indices[ib, jb]
                if index == -1:
                    iblist.append([])
                else:
                    b = b_indices[index]
                    if len(b) > 0:
                        bt = []
                        for bi in b:
                            bt.append(torch.where(tree_indices == bi)[0].item())
                        iblist.append(torch.tensor(bt, device=device))
                    else:
                        iblist.append(b)
            b_indices_new.append(iblist)

        # Aggregate the generated buffers into a dictionary
        tree_buffers = {
            "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": tree_indices,
            "tree_position_ids": tree_position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Move the tensors in the dictionary to the specified device
        tree_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in tree_buffers.items()
        }
        tree_buffers["p_indices"] = p_indices_new
        tree_buffers["b_indices"] = b_indices_new
        return tree_buffers

    def prefix_matching_next_tree_tokens(
        model_input_ids, next_tokens, next_token_scores,
        is_prefilling_phase=False,
        input_token_scores = None,
        prefix_token_sampler=None,
        **kwargs,
    ):
        # all_collected_input_ids should only include the first element of model_inputs['input_ids']
        retrieve_indices = kwargs.get("retrieve_indices", None)
        assert not is_prefilling_phase
        assert next_tokens.shape[1] > retrieve_indices.max()
        next_tree_tokens = next_tokens[0][retrieve_indices]
        next_tree_scores = next_token_scores[0][retrieve_indices]

        input_tree_tokens = model_input_ids[0][retrieve_indices]
        input_tree_scores = input_token_scores[0][retrieve_indices]

        # TODO 此处需要将分布tree化
        if prefix_token_sampler is not None:
            # SpeculativeSampler.__call__
            first_misaligned_input_token_inds, next_tree_tokens, next_tree_scores = prefix_token_sampler(
                draft_tokens = input_tree_tokens,
                advanced_tokens = next_tree_tokens,
                draft_prob = input_tree_scores,
                advanced_prob = next_tree_scores,
                **kwargs,
            )
            # min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
            min_first_misaligned_input_token_index = max(first_misaligned_input_token_inds)
            max_indices = [index for index, value in enumerate(first_misaligned_input_token_inds) if value == min_first_misaligned_input_token_index]
        else:
            first_misaligned_input_token_inds = find_first_misaligned_token_inds(
                input_tree_tokens, next_tree_tokens,
            )
            # min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
            min_first_misaligned_input_token_index = max(first_misaligned_input_token_inds)
            max_indices = [index for index, value in enumerate(first_misaligned_input_token_inds) if value == min_first_misaligned_input_token_index]

        # 选择第一条最长匹配路径为结构
        matched_num = min_first_misaligned_input_token_index
        matched_next_tokens = next_tree_tokens[None, max_indices[0], :matched_num]
        matched_next_scores = next_tree_scores[None, max_indices[0], :matched_num]
        
        # 获得match的retrieve_indices
        matched_retrieve_indices = retrieve_indices[max_indices[0], :matched_num]
        unmatched_next_tokens = next_tokens.clone()
        unmatched_next_scores = next_token_scores.clone()
        
        # 将没有match的token保留
        unmatched_next_tokens[:, matched_retrieve_indices] = -1
        unmatched_next_scores[:, matched_retrieve_indices] = -1

        return matched_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores, matched_retrieve_indices

    # replce
    def replace_based_on_max(d_tv, unmatched_next_tree_tokens, unmatched_next_tree_scores, type_choice= 'min'):
        # 获取每列最大值所在的行索引
        # d_tv_abs = torch.abs(d_tv)
        d_tv_abs = d_tv
        if type_choice == 'min':
            min_row_indices = torch.argmin(d_tv_abs, dim=0)  # 形状为[12]
            min_vals = d_tv_abs.min(dim=0).values
        else:
            min_row_indices = torch.argmax(d_tv_abs, dim=0)  # 形状为[12]
            min_vals = d_tv_abs.max(dim=0).values
        
        # 初始化替换后的张量
        batch_size, seq_len = unmatched_next_tree_tokens.shape
        replaced = torch.zeros_like(unmatched_next_tree_tokens)
        replaced_scores = torch.zeros_like(unmatched_next_tree_scores)
        # flag = [True] * batch_size
        # 对于每一行
        for i in range(batch_size):
            # 对于每个位置
            for j in range(seq_len):
                # 如果是前12列，使用max_row_indices[j]对应的行
                # if unmatched_next_tree_tokens[i, j] != -1 and d_tv_abs[i, j-1] != min_vals[j-1] and flag[i]:
                if unmatched_next_tree_tokens[i, j] != -1 and d_tv_abs[i, j-1] != min_vals[j-1]:
                    replaced[i, j] = unmatched_next_tree_tokens[min_row_indices[j-1], j]
                    replaced_scores[i, j] = unmatched_next_tree_scores[min_row_indices[j-1], j]
                    # flag[i] = False
                # 对于超出的列(第13列)，保持原样
                else:
                    replaced[i, j] = unmatched_next_tree_tokens[i, j]
                    replaced_scores[i, j] = unmatched_next_tree_scores[i, j]
        
        return replaced, replaced_scores

    def prefix_matching_next_tree_tokens_replace(
        model_input_ids, next_tokens, next_token_scores,
        is_prefilling_phase=False,
        input_token_scores = None,
        prefix_token_sampler=None,
        **kwargs,
    ):
        # all_collected_input_ids should only include the first element of model_inputs['input_ids']
        retrieve_indices = kwargs.get("retrieve_indices", None)
        assert not is_prefilling_phase
        assert next_tokens.shape[1] > retrieve_indices.max()
        next_tree_tokens = next_tokens[0][retrieve_indices]
        next_tree_scores = next_token_scores[0][retrieve_indices]

        input_tree_tokens = model_input_ids[0][retrieve_indices]
        input_tree_scores = input_token_scores[0][retrieve_indices]

        # TODO 此处需要将分布tree化
        if prefix_token_sampler is not None:
            # SpeculativeSampler.__call__
            first_misaligned_input_token_inds, next_tree_tokens, next_tree_scores, d_tv = prefix_token_sampler(
                draft_tokens = input_tree_tokens,
                advanced_tokens = next_tree_tokens,
                draft_prob = input_tree_scores,
                advanced_prob = next_tree_scores,
                **kwargs,
            )
            # min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
            min_first_misaligned_input_token_index = max(first_misaligned_input_token_inds)
            max_indices = [index for index, value in enumerate(first_misaligned_input_token_inds) if value == min_first_misaligned_input_token_index]
        else:
            first_misaligned_input_token_inds = find_first_misaligned_token_inds(
                input_tree_tokens, next_tree_tokens,
            )
            # min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
            min_first_misaligned_input_token_index = max(first_misaligned_input_token_inds)
            max_indices = [index for index, value in enumerate(first_misaligned_input_token_inds) if value == min_first_misaligned_input_token_index]

        # 选择第一条最长匹配路径为结构
        matched_num = min_first_misaligned_input_token_index
        matched_next_tokens = next_tree_tokens[None, max_indices[0], :matched_num]
        matched_next_scores = next_tree_scores[None, max_indices[0], :matched_num]
        
        # 获得match的retrieve_indices
        matched_retrieve_indices = retrieve_indices[max_indices[0], :matched_num]
        unmatched_next_tokens = next_tokens.clone()
        unmatched_next_scores = next_token_scores.clone()
        
        # 将没有match的token保留
        unmatched_next_tokens[:, matched_retrieve_indices] = -1
        unmatched_next_scores[:, matched_retrieve_indices] = -1
        
        # 做替换的操作
        unmatched_next_tree_tokens = unmatched_next_tokens[0][retrieve_indices]
        unmatched_next_tree_scores = unmatched_next_scores[0][retrieve_indices]
        unmatched_next_tree_tokens, unmatched_next_tree_scores = replace_based_on_max(d_tv, unmatched_next_tree_tokens, unmatched_next_tree_scores)
        
        # 变换回来
        batch_size, seq_len = 1, unmatched_next_tokens.shape[1]
        flat_indices = retrieve_indices.reshape(-1)       # [90]
        flat_values = unmatched_next_tree_tokens.reshape(1, -1)  # [1, 90]
        flat_values_scores = unmatched_next_tree_scores.reshape(1, -1, unmatched_next_tree_scores.shape[-1])  # [1, 90]
        # init restore tensor
        tree_input_ids_restore = torch.zeros(batch_size, seq_len,
                                            dtype=unmatched_next_tokens.dtype,
                                            device=unmatched_next_tokens.device)
        
        tree_input_scores_restore = torch.zeros(unmatched_next_scores.shape[0], unmatched_next_scores.shape[1], unmatched_next_scores.shape[2],
                                            dtype=unmatched_next_scores.dtype,
                                            device=unmatched_next_scores.device)

        # scatter back
        tree_input_ids_restore.scatter_(1, flat_indices.unsqueeze(0), flat_values)
        tree_input_scores_restore.index_copy_(
            1, 
            flat_indices, 
            flat_values_scores.to(tree_input_scores_restore.dtype)
        )
        return matched_num, matched_next_tokens, tree_input_ids_restore, matched_next_scores, tree_input_scores_restore, matched_retrieve_indices
        # return matched_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores, matched_retrieve_indices

    def delete_false_key_value_tree(
        self,
        output_token_num,
        matched_retrieve_indices,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for layer_idx in range(len(self.key_cache)):
            # 保留tree cache并删除tree cache
            key_tree_cache = self.key_cache[layer_idx][..., -output_token_num:, :]
            value_tree_cache = self.value_cache[layer_idx][..., -output_token_num:, :]
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-output_token_num, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-output_token_num, :]  
            # 提取出accept的token_cache
            key_tree_accept_cache = key_tree_cache[..., matched_retrieve_indices, :]
            value_tree_accept_cache = value_tree_cache[..., matched_retrieve_indices, :]
            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_tree_accept_cache], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_tree_accept_cache], dim=-2)

    def push_forward_model_kwargs_and_inputs_tree(
        model_kwargs,
        all_collected_input_ids, 
        model_input_ids, output_token_num,
        num_matched_tokens, matched_next_tokens,
        temporary_collected_scores=None,
        matched_next_scores=None,
        matched_retrieve_indices=None
    ):

        updated_input_ids = torch.cat([all_collected_input_ids, matched_next_tokens], dim=-1) # all_collected_input_ids 是接受的, matched_next_tokens是匹配成功的
        if temporary_collected_scores is not None:
            temporary_collected_scores = torch.cat([
                temporary_collected_scores[:, -1:], 
                matched_next_scores,
            ], dim=-2)
        
        past_key_values = model_kwargs["past_key_values"]
        attention_mask = model_kwargs["attention_mask"]
        cache_position = model_kwargs["cache_position"]

        new_model_inputs = None

        seq_len = cache_position.shape[-1]
        remaining_tokens_num = seq_len - num_matched_tokens
        if remaining_tokens_num > 0:
            # roll back but in tree way
            delete_false_key_value_tree(past_key_values, output_token_num, matched_retrieve_indices)

            past_attention_mask = attention_mask[..., :num_matched_tokens, :-output_token_num]
            ones_tensor = torch.ones(
                (past_attention_mask.shape[0], past_attention_mask.shape[1], past_attention_mask.shape[1]),
                dtype=past_attention_mask.dtype,
                device=past_attention_mask.device
            )
            lower_triangular = torch.tril(ones_tensor)
            attention_mask = torch.cat([past_attention_mask, lower_triangular], dim=-1)
            cache_position = cache_position[..., matched_retrieve_indices]

            new_model_inputs = {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }

            model_kwargs.update(new_model_inputs)
        return model_kwargs, updated_input_ids, temporary_collected_scores

    class JacobiSampler(model_class, nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()
        
        def prepare_inputs_for_generation_jacobi(
            self,
            input_ids,
            pixel_values=None,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            prefill_num = 3,
            **kwargs,
        ):
            # filling random tokens for multi-next-token prediction
            is_append_random_tokens = kwargs.get("is_append_random_tokens", False) ###!!!
            temporary_collected_scores = kwargs.get("temporary_collected_scores", None)
            tree_input_ids = kwargs.get("tree_input_ids", None)
            tree_input_scores = kwargs.get("tree_input_scores", None)

            input_ids_accept_len_ptr = kwargs.get("input_ids_accept_len_ptr", 0)

            if is_append_random_tokens:
                rand_tokens, scores_of_rand_tokens = get_draft_token_for_tree(
                    img_vocab=self.img_vocab, tree_input_ids = tree_input_ids,
                    input_ids=input_ids, temporary_collected_scores = temporary_collected_scores,
                    device=input_ids.device,
                    multi_token_init_scheme=self.multi_token_init_scheme
                )

                # 找到所有值为-1的位置（布尔掩码）
                insert_mask = (tree_input_ids == -1)
                
                # 将rand_tokens按顺序替换到tree_token的-1位置
                tree_input_ids[insert_mask] = rand_tokens
                tree_input_scores[insert_mask] = scores_of_rand_tokens
            assert tree_input_ids.min() >= 0, f"tree_input_ids: {tree_input_ids}"
            assert tree_input_scores.min() >= 0, f"tree_input_scores: {tree_input_scores}"
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                if len(position_ids.shape) == 3: ###!!!
                    position_ids = torch.max(position_ids, dim=1)[0]
                
                if past_key_values:
                    position_ids = position_ids[:, -tree_input_ids.shape[1] :]

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": tree_input_ids.contiguous()}  # `contiguous()` needed for compilation use cases


            if cache_position[0] == 0:
                # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
                # Otherwise we need pixel values to be passed to model
                model_inputs["pixel_values"] = pixel_values
            
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )

            model_inputs['input_token_scores'] = tree_input_scores
            model_inputs['input_ids_accept_len_ptr'] = input_ids_accept_len_ptr
            model_inputs['rand_tokens_shape'] = rand_tokens.shape[-1]

            return model_inputs

        def prepare_cfg_input(
            self, 
            model_inputs, 
            cfg_repeat_name_list, 
            prefill_num=None,
            neg_input_ids = None,
        ):
            def cfg_repeat(x):
                return x.repeat(2, *([1] * (len(x.shape) - 1)))
            
            for name in cfg_repeat_name_list:
                if (name in model_inputs) and (model_inputs[name] is not None):

                    if name == 'attention_mask':
                        model_inputs[name] = cfg_repeat(model_inputs[name])
                        B = model_inputs[name].shape[0]
                        model_inputs[name][B//2:, :prefill_num] = 0
                    elif name == 'input_ids' and neg_input_ids is not None:
                        input_ids = model_inputs[name]
                        neg_input_ids = neg_input_ids
                        model_inputs[name] = get_double_cfg_input_ids(
                            input_ids, 
                            neg_input_ids,
                            pad_category = self.config.pad_token_id,
                        )
                    else:
                        model_inputs[name] = cfg_repeat(model_inputs[name])
             
            return model_inputs

        def _get_initial_cache_position(self, input_ids, model_kwargs):
            """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
            # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
            if "inputs_embeds" in model_kwargs:
                cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            else:
                cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

            past_length = 0
            if model_kwargs.get("past_key_values") is not None:
                cache = model_kwargs["past_key_values"]
                past_length = 0
                if not isinstance(cache, Cache):
                    past_length = cache[0][0].shape[2]
                elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                    past_length = cache.get_seq_length()

                # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
                # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
                if not is_torchdynamo_compiling():
                    cache_position = cache_position[past_length:]

            model_kwargs["cache_position"] = cache_position

            return model_kwargs

        def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
            tree_choices: list = None,
            is_prefilling_phase: bool = True,
        ) -> Dict[str, Any]:
            assert tree_choices != None
            assert not is_encoder_decoder
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            model_kwargs[cache_name] = cache
            if getattr(outputs, "state", None) is not None:
                model_kwargs["state"] = outputs.state

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            assert ("attention_mask" in model_kwargs)
            if not is_prefilling_phase and not self.init_successful:
                assert num_new_tokens != 1
                attention_mask = model_kwargs["attention_mask"]
                shape_tree_choices = tree_choices
                if num_new_tokens != len(tree_choices) + 1:
                    shape_tree_choices = tree_choices[:num_new_tokens - 1]
                    model_kwargs['tree_input_ids'] = model_kwargs['tree_input_ids'][:, :num_new_tokens]
                    model_kwargs['tree_input_scores'] = model_kwargs['tree_input_scores'][:, :num_new_tokens]
                tree_buffers = generate_tree_buffers(shape_tree_choices, device=cache[0][0].device)
                # update attention mask
                tree_attn_mask = tree_buffers["tree_attn_mask"][0]
                tree_attn_mask = tree_attn_mask.repeat(2, 1, 1)
                attention_mask = attention_mask[:, -1:, :].repeat(1, tree_attn_mask.shape[1], 1) #复制 给每一个 tree node
                model_kwargs["attention_mask"] = torch.cat([attention_mask, tree_attn_mask], dim=-1)
                # update positions_ids
                past_positions = model_kwargs.pop("cache_position")
                new_positions = tree_buffers["tree_position_ids"] + 1 + past_positions[-1]
                model_kwargs["cache_position"] = new_positions.to(past_positions.device)

                model_kwargs["tree_indices"] = tree_buffers["tree_indices"]
                model_kwargs["retrieve_indices"] = tree_buffers["retrieve_indices"]
                model_kwargs["p_indices"] = tree_buffers["p_indices"]
                model_kwargs["b_indices"] = tree_buffers["b_indices"]
                self.init_successful = True
                self.buffers_tree_attn_mask = tree_attn_mask
                self.buffers_tree_position_ids = tree_buffers["tree_position_ids"]
                # self.buffers_tree_indices = tree_buffers["tree_indices"]
                # self.buffers_retrieve_indices = tree_buffers["retrieve_indices"]
                # self.buffers_p_indices = tree_buffers["p_indices"]
                # self.buffers_b_indices = tree_buffers["b_indices"]
            elif not is_prefilling_phase and self.init_successful:
                attention_mask = model_kwargs["attention_mask"]
                shape_tree_choices = tree_choices
                # must do
                if num_new_tokens != len(tree_choices) + 1:
                    shape_tree_choices = tree_choices[:num_new_tokens - 1]
                    model_kwargs['tree_input_ids'] = model_kwargs['tree_input_ids'][:, :num_new_tokens]
                    model_kwargs['tree_input_scores'] = model_kwargs['tree_input_scores'][:, :num_new_tokens]
                tree_attn_mask = self.buffers_tree_attn_mask
                attention_mask = attention_mask[:, -1:, :].repeat(1, tree_attn_mask.shape[1], 1) #复制 给每一个 tree node
                model_kwargs["attention_mask"] = torch.cat([attention_mask, tree_attn_mask], dim=-1)
                
                # update positions_ids
                past_positions = model_kwargs.pop("cache_position")
                new_positions = self.buffers_tree_position_ids + 1 + past_positions[-1]
                model_kwargs["cache_position"] = new_positions.to(past_positions.device)
            else:
                self.init_successful = False
                assert num_new_tokens == 1
                # update attention mask
                attention_mask = model_kwargs["attention_mask"]

                while len(attention_mask.shape) < 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                attention_mask = attention_mask[..., -1:, :] #
                
                if attention_mask.shape[-2] < num_new_tokens:
                    attention_mask = attention_mask.expand(
                        (attention_mask.shape[0], num_new_tokens, attention_mask.shape[-1])
                    )
                
                model_kwargs["attention_mask"] = torch.ones(
                    ( attention_mask.shape[0], num_new_tokens, num_new_tokens + attention_mask.shape[-1] ),
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                model_kwargs["attention_mask"][..., :, :attention_mask.shape[-1]] = attention_mask[..., -1:, :]
                model_kwargs["attention_mask"][..., :, attention_mask.shape[-1]:] = torch.tril(
                    model_kwargs["attention_mask"][0, :, attention_mask.shape[-1]:]
                )
                # update positions_ids
                if model_kwargs.get("use_cache", True):
                    past_positions = model_kwargs.pop("cache_position")
                    new_positions = torch.arange(
                        past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                    ).to(past_positions.device)
                    model_kwargs["cache_position"] = new_positions
                else:
                    past_positions = model_kwargs.pop("cache_position")
                    new_positions = torch.arange(
                        past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                    ).to(past_positions.device)
                    model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            
            return model_kwargs

        def _init_new_params(
            self, 
            jacobi_loop_interval_l = 1,
            jacobi_loop_interval_r = (768 // 16)**2 + 768 // 16, # This should be determined by the image size ###!!!
            max_num_new_tokens = 16,
            guidance_scale = 3.0,
            seed = 42,
            multi_token_init_scheme = 'random',
            do_cfg = True,
            prefix_token_sampler_scheme = 'speculative_jacobi',
            use_chameleon_tokenizer = True,
            _init_doubled_attn_mask_cfg = False,
            **kwargs,
        ):
            local_chameleon_tokenizer_path = kwargs.get("local_chameleon_tokenizer_path", None)
            if use_chameleon_tokenizer:
                import model.chameleon_vae_ori as chameleon_vae_ori
                import os
                assert os.path.exists(local_chameleon_tokenizer_path), f"{local_chameleon_tokenizer_path}"
                chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
                    json.load(open(f"{local_chameleon_tokenizer_path}/text_tokenizer.json"))["model"]["vocab"]
                )
                chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(chameleon_ori_vocab)
                img_vocab = chameleon_ori_translation._vocab.image_tokens
                self.register_buffer("img_vocab", torch.tensor(img_vocab, dtype=torch.long))
            else:
                if not hasattr(self, 'img_vocab'):
                    self.img_vocab = None

            self.cfg_repeat_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 
            ]
            self.cfg_half_name_list = [
                'inputs_embeds', 'input_ids', 'pixel_values', 
            ]
            self.jacobi_loop_interval_l = jacobi_loop_interval_l
            self.jacobi_loop_interval_r = jacobi_loop_interval_r
            self.max_num_new_tokens = max_num_new_tokens
            self.max_jacobi_iter_num = min(200, self.max_num_new_tokens+1) ###!!!
            self.guidance_scale = guidance_scale

            self.seed = seed
            self.generator = None

            self.multi_token_init_scheme = multi_token_init_scheme
            self.do_cfg = do_cfg

            self.prefix_token_sampler_scheme = prefix_token_sampler_scheme
            self._init_doubled_attn_mask_cfg = _init_doubled_attn_mask_cfg
        
        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                logits_warper (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                    to warp the prediction score distribution of the language modeling head applied before multinomial
                    sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                    `generation_config`)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            # assert False, f"pad_token_id: {pad_token_id}"
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            if hasattr(generation_config, 'return_accl'):
                return_accl = generation_config.return_accl
            else:
                return_accl = False
            # ---- tree sampler args
            tree_choices = generation_config.tree_choices

            max_length = generation_config.max_length
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            init_len = cur_len
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            
            temporary_collected_scores = input_ids.new_zeros((batch_size, cur_len, self.config.vocab_size))
            temporary_collected_scores = torch.scatter(temporary_collected_scores, 2, input_ids.unsqueeze(-1), 1.0)

            # init: attn mask, cache_position, cfg, 
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
            prefill_num = model_kwargs['attention_mask'].shape[1] - 1

            do_cfg = self.do_cfg if hasattr(self, 'do_cfg') else False

            guidance_scale = self.guidance_scale if hasattr(self, 'guidance_scale') else 3.0
            do_cfg = (do_cfg & (guidance_scale != 1))
            
            if do_cfg:
                model_kwargs = self.prepare_cfg_input(
                    model_kwargs, 
                    cfg_repeat_name_list = ['attention_mask', ] if (
                        not self._init_doubled_attn_mask_cfg
                    ) else [],
                    prefill_num = prefill_num ,
                )

            # prefilling tokens always output 1 next token
            output_token_num = 1
            additional_tokens = None
            additional_scores = None

            if self.seed is not None:
                set_seed(self.seed)
                self.generator = torch.Generator(input_ids.device).manual_seed(self.seed)

            jacobi_loop_interval_lr = (cur_len + self.jacobi_loop_interval_l, cur_len + self.jacobi_loop_interval_r)
            gen_loop_num = 0

            device = input_ids.device
            dtype = input_ids.dtype

            #---------------Setting sampling-----------------
            sequence_K = 0
            for single_node in tree_choices:
                if len(single_node) == 1:
                    sequence_K += 1
            if self.prefix_token_sampler_scheme == 'speculative_jacobi':
                prefix_token_sampler = SpeculativeSampler(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        batch_size, dtype=dtype, device=device
                    ),
                )
            elif self.prefix_token_sampler_scheme == 'jacobi':
                prefix_token_sampler = None
            elif self.prefix_token_sampler_scheme == 'lantern':
                if hasattr(generation_config, 'lantern_delta'):
                    lantern_delta = generation_config.lantern_delta
                else:
                    raise ValueError("lantern_delta must be specified in generation_config")
                prefix_token_sampler = SpeculativeSamplerLantern(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        batch_size, dtype=dtype, device=device
                    ),
                    lantern_delta = lantern_delta
                )
            elif self.prefix_token_sampler_scheme == 'group_speculative_jacobi':
                head_img_weight = self.model.embed_tokens.weight[:10000]
                a = head_img_weight
                head_img_sims = (torch.sum(a**2, dim=-1, keepdim=True) + torch.sum(a.T**2,dim =0, keepdim=True)) - 2*a@a.T
                prefix_token_sampler = GroupSpeculativeSampler(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        batch_size, dtype=dtype, device=device
                    ),
                )
            
                prefix_token_sampler.head_img_sims = head_img_sims
            elif self.prefix_token_sampler_scheme == 'speculative_jacobi_groupsum':
                if hasattr(generation_config, 'groupsum_delta'):
                    groupsum_delta = generation_config.groupsum_delta
                else:
                    raise ValueError("groupsum_delta must be specified in generation_config")
                prefix_token_sampler = SpeculativeGroupSumSampler(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        batch_size, dtype=dtype, device=device
                    ),
                    groupsum_delta = groupsum_delta
                )
                prefix_token_sampler.sequence_K = sequence_K
            elif self.prefix_token_sampler_scheme == 'group_speculative_jacobi_groupsum':
                if hasattr(generation_config, 'groupsum_delta'):
                    groupsum_delta = generation_config.groupsum_delta
                else:
                    raise ValueError("groupsum_delta must be specified in generation_config")
                head_img_weight = self.model.embed_tokens.weight[:10000]
                a = head_img_weight
                head_img_sims = (torch.sum(a**2, dim=-1, keepdim=True) + torch.sum(a.T**2,dim =0, keepdim=True)) - 2*a@a.T
                prefix_token_sampler = GroupSpeculativeGroupSumSampler(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        batch_size, dtype=dtype, device=device
                    ),
                    groupsum_delta = groupsum_delta
                )
                
                prefix_token_sampler.sequence_K = sequence_K
                prefix_token_sampler.head_img_sims = head_img_sims
            else:
                raise ValueError(f"prefix_token_sampler_scheme: {self.prefix_token_sampler_scheme}")

            count_time = True
            if count_time:
                t1 = torch.cuda.Event(enable_timing=True)
                t2 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t1.record()

            #---------------Init for tree-----------------
            model_kwargs['tree_input_ids'] = input_ids
            model_kwargs['tree_input_scores'] = temporary_collected_scores
            max_num_new_tokens = len(tree_choices) + 1

            #---------------LOOP for Init-----------------
            while 1 == output_token_num:
                # prepare model inputs
                # place the last `len(cache_position)` elements of `input_ids` into `model_kwargs` 
                model_inputs = self.prepare_inputs_for_generation_jacobi(
                    input_ids, 
                    num_fill_rand_tokens = output_token_num - 1,
                    is_append_random_tokens = True,
                    additional_tokens = additional_tokens,
                    additional_scores = additional_scores,
                    temporary_collected_scores = temporary_collected_scores,
                    img_width = logits_processor[0].w_latent_dim if hasattr(logits_processor[0], 'w_latent_dim') else None,
                    generator = self.generator,
                    prefill_num = prefill_num,
                    **model_kwargs,
                )
                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # the first element of model_inputs['input_ids'] is in all_collected_input_ids
                all_collected_input_ids = input_ids
                model_input_ids = model_inputs['input_ids']
                input_token_scores = model_inputs.pop('input_token_scores')
                input_ids_accept_len_ptr = model_inputs.pop('input_ids_accept_len_ptr')
                rand_tokens_shape = model_inputs.pop('rand_tokens_shape')

                is_force_no_cfg = check_is_force_no_cfg(
                    input_ids, 
                    image_start_token_id = logits_processor[0].image_start_token_id if hasattr(
                        logits_processor[0], 'image_start_token_id'
                    ) else None,
                    image_end_token_id = logits_processor[0].image_end_token_id if hasattr(
                        logits_processor[0], 'image_end_token_id'
                    ) else None,
                    guidance_scale=guidance_scale,
                    do_cfg = do_cfg,
                ) # to adapt the bs=2 kv cache
                if do_cfg:
                    model_inputs = self.prepare_cfg_input(
                        model_inputs,
                        cfg_repeat_name_list = self.cfg_repeat_name_list,
                        neg_input_ids = model_kwargs.get(
                            'neg_input_ids', None
                        ) if (gen_loop_num == 0) else None,
                    )
                
                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                logits = outputs.logits
                next_tokens, next_token_scores = sampling_logits2tokens(
                    logits,
                    all_collected_input_ids,
                    unfinished_sequences, pad_token_id,
                    output_token_num = output_token_num,
                    logits_processor=logits_processor, logits_warper=logits_warper,
                    do_sample=do_sample,
                    has_eos_stopping_criteria=has_eos_stopping_criteria,
                    do_cfg=do_cfg,
                    generator=self.generator, # token_sampler = prefix_token_sampler,
                    guidance_scale=guidance_scale,
                    is_force_no_cfg=is_force_no_cfg,
                )

                if do_cfg:
                    model_inputs = postprocess_cfg_decode(model_inputs)
                
                # verify phase
                num_matched_tokens, matched_next_tokens, unmatched_next_tokens, \
                matched_next_scores, unmatched_next_scores = prefix_matching_next_tokens(
                    model_input_ids=model_input_ids, 
                    input_token_scores = input_token_scores,
                    next_tokens=next_tokens, 
                    next_token_scores=next_token_scores,
                    is_prefilling_phase = (output_token_num <= 1),
                    prefix_token_sampler = prefix_token_sampler,
                    logits_processor = logits_processor, logits_warper = logits_warper,
                    all_collected_input_ids = all_collected_input_ids,
                )

                output_token_num = min(max_num_new_tokens, jacobi_loop_interval_lr[-1] - cur_len) if (
                    cur_len >= jacobi_loop_interval_lr[0]
                ) and (cur_len < jacobi_loop_interval_lr[-1]) else 1

                model_kwargs, updated_input_ids, \
                nonoverlap_output_token_num, \
                additional_tokens, additional_scores, \
                temporary_collected_scores = push_forward_model_kwargs_and_inputs(
                    model_kwargs=model_kwargs, 
                    all_collected_input_ids=all_collected_input_ids, 
                    model_input_ids=model_input_ids, 
                    output_token_num=output_token_num,
                    num_matched_tokens=num_matched_tokens,
                    matched_next_tokens=matched_next_tokens,
                    unmatched_next_tokens=unmatched_next_tokens,
                    temporary_collected_scores=temporary_collected_scores,
                    matched_next_scores=matched_next_scores,
                    unmatched_next_scores=unmatched_next_scores,
                )
                input_ids = updated_input_ids
                model_kwargs['tree_input_ids'] = matched_next_tokens[:, -1:]
                model_kwargs['tree_input_scores'] = matched_next_scores[:, -1:]

                # Store scores, attentions and hidden_states when required
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs
                if streamer is not None:
                    if len(next_tokens.shape) == 1:
                        streamer.put(next_tokens.cpu())
                    else:
                        for j in range(next_tokens.shape[1]):
                            streamer.put(next_tokens[:, j].cpu())
                
                # 更新attention mask, position ids, past key values
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens = nonoverlap_output_token_num,
                    is_prefilling_phase = (output_token_num <= 1),
                    tree_choices = tree_choices,
                )

                # check whether we get the end token
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

                cur_len = input_ids.shape[1]
                gen_loop_num += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs
            #---------------Init for Normal-----------------
            pad_tree_token = torch.full((model_kwargs['tree_input_ids'].shape[0],max_num_new_tokens-1), -1, dtype=model_kwargs['tree_input_ids'].dtype, device=model_kwargs['tree_input_ids'].device)
            pad_tree_scores = torch.full((model_kwargs['tree_input_ids'].shape[0],max_num_new_tokens-1, self.config.vocab_size), -1, dtype=temporary_collected_scores.dtype, device=temporary_collected_scores.device)
            model_kwargs['tree_input_ids'] = torch.cat([model_kwargs['tree_input_ids'], pad_tree_token], dim=1)
            model_kwargs['tree_input_scores'] = torch.cat([model_kwargs['tree_input_scores'], pad_tree_scores], dim=1)
            #---------------LOOP for Normal-----------------
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length):
                # prepare model inputs
                # place the last `len(cache_position)` elements of `input_ids` into `model_kwargs`
                model_inputs = self.prepare_inputs_for_generation_jacobi(
                    input_ids, 
                    # num_fill_rand_tokens = output_token_num - 1,
                    is_append_random_tokens = True,
                    additional_tokens = additional_tokens,
                    additional_scores = additional_scores,
                    temporary_collected_scores = temporary_collected_scores,
                    img_width = logits_processor[0].w_latent_dim if hasattr(logits_processor[0], 'w_latent_dim') else None,
                    generator = self.generator,
                    prefill_num = prefill_num,
                    **model_kwargs,
                )
                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # the first element of model_inputs['input_ids'] is in all_collected_input_ids
                all_collected_input_ids = input_ids
                model_input_ids = model_inputs['input_ids']

                input_token_scores = model_inputs.pop('input_token_scores')
                input_ids_accept_len_ptr = model_inputs.pop('input_ids_accept_len_ptr')
                rand_tokens_shape = model_inputs.pop('rand_tokens_shape')

                is_force_no_cfg = check_is_force_no_cfg(
                    input_ids, 
                    image_start_token_id = logits_processor[0].image_start_token_id if hasattr(
                        logits_processor[0], 'image_start_token_id'
                    ) else None,
                    image_end_token_id = logits_processor[0].image_end_token_id if hasattr(
                        logits_processor[0], 'image_end_token_id'
                    ) else None,
                    guidance_scale=guidance_scale,
                    do_cfg = do_cfg,
                ) # to adapt the bs=2 kv cache
                if do_cfg:
                    model_inputs = self.prepare_cfg_input(
                        model_inputs,
                        cfg_repeat_name_list = self.cfg_repeat_name_list,
                        neg_input_ids = model_kwargs.get(
                            'neg_input_ids', None
                        ) if (gen_loop_num == 0) else None,
                    )
                
                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                logits = outputs.logits
                next_tokens, next_token_scores = sampling_logits2tokens(
                    logits,
                    all_collected_input_ids,
                    unfinished_sequences, pad_token_id,
                    output_token_num = output_token_num,
                    logits_processor=logits_processor, logits_warper=logits_warper,
                    do_sample=do_sample,
                    has_eos_stopping_criteria=has_eos_stopping_criteria,
                    do_cfg=do_cfg,
                    generator=self.generator, # token_sampler = prefix_token_sampler,
                    guidance_scale=guidance_scale,
                    is_force_no_cfg=is_force_no_cfg,
                    cache_position = model_kwargs['cache_position'],
                )

                if do_cfg:
                    model_inputs = postprocess_cfg_decode(model_inputs)
                # verify tree phase
                if self.prefix_token_sampler_scheme in ['speculative_jacobi_replace', 'lantern_replace', 'group_speculative_jacobi_replace']:
                    num_matched_tokens, matched_next_tokens, unmatched_next_tokens, \
                    matched_next_scores, unmatched_next_scores, matched_retrieve_indices = prefix_matching_next_tree_tokens_replace(
                        model_input_ids=model_input_ids, 
                        input_token_scores = input_token_scores,
                        next_tokens=next_tokens, 
                        next_token_scores=next_token_scores,
                        is_prefilling_phase = (output_token_num <= 1),
                        prefix_token_sampler = prefix_token_sampler,
                        logits_processor = logits_processor, logits_warper = logits_warper,
                        all_collected_input_ids = all_collected_input_ids,
                        # Tree
                        retrieve_indices = model_kwargs['retrieve_indices'],
                    )
                else:
                    num_matched_tokens, matched_next_tokens, unmatched_next_tokens, \
                    matched_next_scores, unmatched_next_scores, matched_retrieve_indices = prefix_matching_next_tree_tokens(
                        model_input_ids=model_input_ids, 
                        input_token_scores = input_token_scores,
                        next_tokens=next_tokens, 
                        next_token_scores=next_token_scores,
                        is_prefilling_phase = (output_token_num <= 1),
                        prefix_token_sampler = prefix_token_sampler,
                        logits_processor = logits_processor, logits_warper = logits_warper,
                        all_collected_input_ids = all_collected_input_ids,
                        # Tree
                        retrieve_indices = model_kwargs['retrieve_indices'],
                    )

                this_time_len = output_token_num
                max_limit_len = jacobi_loop_interval_lr[-1] - cur_len
                # output_token_num = min(max_num_new_tokens, max_limit_len) if (
                #     cur_len >= jacobi_loop_interval_lr[0]
                # ) and (cur_len < jacobi_loop_interval_lr[-1]) else 1
                output_token_num = max_num_new_tokens

                model_kwargs, updated_input_ids, \
                temporary_collected_scores = push_forward_model_kwargs_and_inputs_tree(
                    model_kwargs=model_kwargs, 
                    all_collected_input_ids=all_collected_input_ids, 
                    model_input_ids=model_input_ids, 
                    output_token_num=this_time_len,
                    num_matched_tokens=num_matched_tokens,
                    matched_next_tokens=matched_next_tokens,
                    temporary_collected_scores=temporary_collected_scores,
                    matched_next_scores=matched_next_scores,
                    matched_retrieve_indices = matched_retrieve_indices
                )
                
                nonoverlap_output_token_num = output_token_num
                input_ids = updated_input_ids
                #------------重新构建tree_tokens------------
                tree_input_ids = unmatched_next_tokens.clone()
                tree_input_scores = unmatched_next_scores.clone()
                # 取最后一个位置放入新采样的token
                tree_input_ids[:, 0] = matched_next_tokens[:,-1]
                tree_input_scores[:, 0] = matched_next_scores[:,-1]
                
                # 将-1的那一个group补齐
                retrieve_indices = model_kwargs['retrieve_indices']
                tree_input_ids_line = tree_input_ids[:, retrieve_indices][0]
                tree_input_scores_line = tree_input_scores[:, retrieve_indices][0]
                rand_num = 0
                for i in range(tree_input_ids_line.shape[0]):
                    rand_num = max(rand_num, (tree_input_ids_line[i] == -1).sum())
                #------------删掉等候区的无用的draft token------------
                if rand_num > 0:
                    tree_input_ids_line[:,1:] = torch.cat([tree_input_ids_line[:, 1:][:, rand_num:],tree_input_ids_line[:,1:][:, :rand_num]], dim=1)
                    tree_input_scores_line[:,1:] = torch.cat([tree_input_scores_line[:, 1:][:, rand_num:],tree_input_scores_line[:,1:][:, :rand_num]], dim=1)
                    tree_input_ids_line[:,1:][:, -rand_num:] = -1
                
                # flatten
                batch_size, seq_len = 1, tree_input_ids.shape[1]
                flat_indices = retrieve_indices.reshape(-1)       # [90]
                flat_values = tree_input_ids_line.reshape(1, -1)  # [1, 90]
                flat_values_scores = tree_input_scores_line.reshape(1, -1, tree_input_scores.shape[-1])  # [1, 90]
                # init restore tensor
                tree_input_ids_restore = torch.zeros(batch_size, seq_len,
                                                    dtype=tree_input_ids.dtype,
                                                    device=tree_input_ids.device)
                
                tree_input_scores_restore = torch.zeros(tree_input_scores.shape[0], tree_input_scores.shape[1], tree_input_scores.shape[2],
                                                    dtype=tree_input_scores.dtype,
                                                    device=tree_input_scores.device)

                # scatter back
                tree_input_ids_restore.scatter_(1, flat_indices.unsqueeze(0), flat_values)
                tree_input_scores_restore.index_copy_(
                    1, 
                    flat_indices, 
                    flat_values_scores.to(tree_input_scores_restore.dtype)
                )
                tree_input_ids = tree_input_ids_restore
                tree_input_scores = tree_input_scores_restore

                model_kwargs['tree_input_ids'] = tree_input_ids
                model_kwargs['tree_input_scores'] = tree_input_scores
                #------------tree_tokens构建结束------------

                # Store scores, attentions and hidden_states when required
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs
                if streamer is not None:
                    if len(next_tokens.shape) == 1:
                        streamer.put(next_tokens.cpu())
                    else:
                        for j in range(next_tokens.shape[1]):
                            streamer.put(next_tokens[:, j].cpu())
                
                # 更新attention mask, position ids, past key values
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens = nonoverlap_output_token_num,
                    is_prefilling_phase = (output_token_num <= 1),
                    tree_choices = tree_choices,
                )

                # check whether we get the end token
                # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                for i in range(cur_len, input_ids.shape[1]+1):
                    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids[:, :i], scores)
                this_peer_finished = unfinished_sequences.max() == 0

                cur_len = input_ids.shape[1]
                gen_loop_num += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            #---------------Return---------------
            if streamer is not None:
                streamer.end()
            
            if count_time:
                t2.record()
                torch.cuda.synchronize()

                t = t1.elapsed_time(t2) / 1000

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            elif return_accl:
                return Result(input_ids=input_ids, loop_num=gen_loop_num, token_gen_len = cur_len - init_len,time_forward=t)
            else:
                return input_ids
    
    return JacobiSampler

def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)
    pipe_line.model._init_new_params(**kwargs)
    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)
    return pipe_line