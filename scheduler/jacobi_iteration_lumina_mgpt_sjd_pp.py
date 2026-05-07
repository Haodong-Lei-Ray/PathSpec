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

from .logit_processor_3dim import MultiTokensVLLogitsProcessor, MultiTokensInterleavedTopKLogitsWarper, \
    get_double_cfg_input_ids, gather_from_split_tensors

from .oneline_utils import get_multi_token_for_preparation
from .oneline_utils import SpeculativeSampler_PlusPlus, SpeculativeSampler_MaxCoupling, SpeculativeSampler_ContinuebyACCl, SpeculativeSampler_ContinuebyACCl_PlusPlus
from .oneline_utils import SpeculativeSampler_RelaxedbyACCl

from absl import logging
import time

class Result:
    def __init__(self, input_ids, loop_num, token_gen_len, time_forward):
        self.input_ids=input_ids
        self.loop_num=loop_num
        self.token_gen_len=token_gen_len
        self.time_forward=time_forward

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def delete_false_key_value(
    self,
    num_of_false_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:

    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

def postprocess_cfg_decode(
    model_inputs,
    cfg_half_name_list=['inputs_embeds', 'input_ids', 'pixel_values', ],
):
    cfg_half_name_list = cfg_half_name_list
    def cfg_half(x):
        return x[:x.shape[0]//2]
    
    for name in cfg_half_name_list:
        if (name in model_inputs) and (model_inputs[name] is not None):
            model_inputs[name] = cfg_half(model_inputs[name])
    
    return model_inputs

def check_is_force_no_cfg(input_ids, image_start_token_id=None, image_end_token_id=None, guidance_scale=3., do_cfg=True):
    if (image_start_token_id is None) or (image_end_token_id is None):
        return False
    
    num_image_start_tokens = (input_ids[0] == image_start_token_id).sum()
    num_image_end_tokens = (input_ids[0] == image_end_token_id).sum()

    if num_image_start_tokens == num_image_end_tokens:
        return True
    else:
        return False

def sampling_logits2tokens(
    logits,
    all_collected_input_ids,
    unfinished_sequences, pad_token_id,
    output_token_num = 1,
    logits_processor=None, logits_warper=None,
    do_sample=True,
    has_eos_stopping_criteria=True,
    do_cfg=False,
    guidance_scale=3.,
    generator=None, #token_sampler = None,
    is_force_no_cfg = False,
):
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = logits[ :, -output_token_num:, : ].clone()

    if do_cfg:
        conditional_logits, unconditional_logits = next_token_logits.chunk(2, dim=0)
        if is_force_no_cfg:
            next_token_logits = conditional_logits
        else:
            next_token_logits = guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits
    
    next_token_scores = logits_processor(all_collected_input_ids, next_token_logits)
    if do_sample and (logits_warper is not None):
        next_token_scores = logits_warper(all_collected_input_ids, next_token_scores)

    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
        probs_shape = None
        if len(probs.shape) >= 3:
            probs_shape = probs.shape
            probs = probs.flatten(0, len(probs_shape)-2)

        next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if probs_shape is not None:
            next_tokens = next_tokens.reshape(probs_shape[:-1])
            probs = probs.reshape(probs_shape)

        next_token_scores = probs
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        next_token_scores = nn.functional.softmax(next_token_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if has_eos_stopping_criteria:
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    
    return next_tokens, next_token_scores

class SpeculativeSampler:

    def __init__(
        self, 
        collected_draft_logits=[], 
        collected_advanced_logits=[], 
        max_num_collected_logits=2,
        generator=None,
        draft_type = 'jacobian_states',
        reject_sampling_relative_ids = None,
        reject_sampling_draft_token_logits = None,
        sampling_last_draft_token = None,
    ):

        self.max_num_collected_logits = max_num_collected_logits
        self.collected_draft_logits = collected_draft_logits
        self.collected_advanced_logits = collected_advanced_logits

        self.draft_token_index_selector = lambda x: x
        if draft_type == 'jacobian_states':
            # for jacobi iteration (predict next token)
            self.advanced_token_index_selector = lambda x: x - 1
        else:
            self.advanced_token_index_selector = lambda x: x

        self.generator = generator

        self.image_token_list = [i for i in range(4, 8195 + 1)] #8192 [4, .., 8195]

        self.reject_sampling_relative_ids = reject_sampling_relative_ids
        self.reject_sampling_draft_token_logits = reject_sampling_draft_token_logits #torch.Size([1, 65536])
        self.sampling_last_draft_token = sampling_last_draft_token

        self._init_reject_sampling_params()
    
    def collect_logits(self, logits, collection_type='draft'):
        if collection_type == 'draft': 
            collected_logits = self.collected_draft_logits
        elif collection_type == 'advanced':
            collected_logits = self.collected_advanced_logits
        else:
            assert False, f"collection_type should be 'draft' or 'advanced', but got {collection_type}"

        if logits is not None:
            collected_logits.append(logits)
        
        if len(collected_logits) > self.max_num_collected_logits:
            return collected_logits.pop(0)
        else:
            return None
    
    def logits_calibrating(self, advanced_prob,):

        calibrated_logits = advanced_prob.log()

        B, L = advanced_prob.shape[:2]
        for b in range(B):
            reject_sampling_relative_index = self.reject_sampling_relative_ids[b]
            reject_sampling_draft_token_logits = self.reject_sampling_draft_token_logits[b]
            if reject_sampling_relative_index >= 0:
                token_advanced_prob = advanced_prob[b, reject_sampling_relative_index]

                calibrated_logits[b, reject_sampling_relative_index] = self.get_reject_sampling_logits(
                    token_advanced_prob, reject_sampling_draft_token_logits)
        
        self._init_reject_sampling_params()

        return calibrated_logits
    
    def get_reject_sampling_logits(self, token_advanced_prob, token_draft_prob):
        pos_delta_logits = (
            token_advanced_prob - token_draft_prob
        ).clamp(min=0).log()
        return pos_delta_logits
    
    def reject_sampling_single_token(
        self, token_advanced_prob, token_draft_prob,
        logits_processor=None, logits_warper=None,
        all_collected_input_ids=None,
    ):

        pos_delta_logits = self.get_reject_sampling_logits(token_advanced_prob, token_draft_prob)
        shape_pos_delta_logits = pos_delta_logits.shape

        if (logits_processor is not None) or (logits_warper is not None):
            while len(all_collected_input_ids.shape) < 2:
                all_collected_input_ids = all_collected_input_ids.unsqueeze(0)
            
            while len(pos_delta_logits.shape) < 3:
                pos_delta_logits = pos_delta_logits.unsqueeze(0)
        
        if logits_processor is not None:
            pos_delta_logits = logits_processor(all_collected_input_ids, pos_delta_logits)
        
        if logits_warper is not None:
            pos_delta_logits = logits_warper(all_collected_input_ids, pos_delta_logits)

        pos_delta_logits = pos_delta_logits.view(shape_pos_delta_logits)
        probs = F.softmax(pos_delta_logits, dim=-1)
        resampled_scores = probs

        probs = probs.unsqueeze(0) if len(probs.shape) <= 1 else probs

        resampled_tokens = torch.multinomial(
            probs, num_samples=1, #len(probs.shape)-1,
            generator=self.generator,
        ).squeeze(-1)
        return resampled_tokens, resampled_scores
        
    def _init_reject_sampling_params(self,):
        self.reject_sampling_relative_ids.fill_(-1)
        self.reject_sampling_draft_token_logits.fill_(0)
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        attn_softmax = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params()

        B, L = draft_tokens.shape

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        for b in range(B):
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # if attn_softmax is not None:
                    # weight = -attn_softmax[b] + 1.8
                    # weight = attn_softmax[b]
                # else:
                weight = 1
                # if r < (sampled_advanced_prob * weight / sampled_draft_prob).clamp(max=1): 
                if r < (sampled_advanced_prob  / (sampled_draft_prob * weight)).clamp(max=1): 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:

                    first_misaligned_token_index = i
                    self.reject_sampling_relative_ids[b] = 0
                    self.reject_sampling_draft_token_logits[b] = draft_prob[b, draft_token_index]

                    # we perform reject sampling in the backbone model's prediction loop out of the this sampler
                    resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                        token_advanced_prob = advanced_prob[b, target_token_index, :], 
                        token_draft_prob = draft_prob[b, draft_token_index, :],
                        logits_processor = logits_processor,
                        logits_warper = logits_warper,
                        all_collected_input_ids = torch.cat([
                            all_collected_input_ids[b, :],
                            resampled_target_tokens[b, :target_token_index],
                        ], dim=-1),
                    )
                    resampled_target_tokens[b, target_token_index] = resampled_tokens
                    # resampled_target_scores[b, target_token_index, :] = resampled_scores # the score is kept, so not to update this.
                    first_misaligned_token_index = i

                    break
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

def find_first_misaligned_token_inds(
    input_ids, next_tokens,
):
    # input_ids: [B, L], next_tokens: [B, L]
    first_misaligned_token_inds = []
    for b in range(input_ids.shape[0]):
        first_misaligned_token_index = input_ids.shape[1] #- 1 # keep at least one token left
        for i in range(1, input_ids.shape[1]):
            if input_ids[b, i] == next_tokens[b, i-1]:
                pass
            else:
                first_misaligned_token_index = i
                break
    
        first_misaligned_token_inds.append(first_misaligned_token_index)
    
    return first_misaligned_token_inds

def prefix_matching_next_tokens(
    model_input_ids, next_tokens, next_token_scores,
    is_prefilling_phase=False,
    input_token_scores = None,
    prefix_token_sampler=None,
    attn_softmax = None,
    **kwargs,
):
    # all_collected_input_ids should only include the first element of model_inputs['input_ids']

    if is_prefilling_phase:
        matched_num = model_input_ids.shape[1]
        matched_next_tokens = next_tokens[:, -1:]
        unmatched_next_tokens = next_tokens[:, next_tokens.shape[1]:]

        matched_next_scores = next_token_scores[:, -1:]
        unmatched_next_scores = next_token_scores[:, next_token_scores.shape[1]:]
    else:

        if prefix_token_sampler is not None:
            # SpeculativeSampler.__call__
            first_misaligned_input_token_inds, next_tokens, next_token_scores = prefix_token_sampler(
                draft_tokens = model_input_ids,
                advanced_tokens = next_tokens,
                draft_prob = input_token_scores,
                advanced_prob = next_token_scores,
                attn_softmax = attn_softmax,
                **kwargs,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)
        else:
            first_misaligned_input_token_inds = find_first_misaligned_token_inds(
                model_input_ids, next_tokens,
            )
            min_first_misaligned_input_token_index = min(first_misaligned_input_token_inds)

        matched_num = min_first_misaligned_input_token_index
        matched_next_tokens = next_tokens[:, :matched_num]
        unmatched_next_tokens = next_tokens[:, matched_num:]

        matched_next_scores = next_token_scores[:, :matched_num]
        unmatched_next_scores = next_token_scores[:, matched_num:]

    return matched_num, matched_next_tokens, unmatched_next_tokens, matched_next_scores, unmatched_next_scores

def push_forward_model_kwargs_and_inputs(
    model_kwargs,
    all_collected_input_ids, 
    model_input_ids, output_token_num,
    num_matched_tokens, matched_next_tokens, unmatched_next_tokens,
    temporary_collected_scores=None,
    matched_next_scores=None,
    unmatched_next_scores=None,
):

    updated_input_ids = torch.cat([all_collected_input_ids, matched_next_tokens], dim=-1) # all_collected_input_ids 是接受的, matched_next_tokens是匹配成功的
    #BUG
    if temporary_collected_scores is not None:
        temporary_collected_scores = torch.cat([
            temporary_collected_scores, 
            matched_next_scores,
        ], dim=-2)
        # temporary_collected_scores = torch.cat([
        #     temporary_collected_scores[:, -1:], 
        #     matched_next_scores,
        # ], dim=-2)
    
    past_key_values = model_kwargs["past_key_values"]
    attention_mask = model_kwargs["attention_mask"]
    cache_position = model_kwargs["cache_position"]

    new_model_inputs = None

    seq_len = cache_position.shape[-1]
    remaining_tokens_num = seq_len - num_matched_tokens
    if remaining_tokens_num > 0:
        # roll back

        delete_false_key_value(past_key_values, remaining_tokens_num)

        attention_mask = attention_mask[..., :-remaining_tokens_num, :-remaining_tokens_num]
        cache_position = cache_position[..., :-remaining_tokens_num]

        new_model_inputs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }

        model_kwargs.update(new_model_inputs)

        additional_tokens = unmatched_next_tokens
        additional_scores = unmatched_next_scores

        nonoverlap_output_token_num = output_token_num 
    else:

        # attention_mask is all useful
        nonoverlap_output_token_num = output_token_num
        additional_tokens = None
        additional_scores = None
    
    return model_kwargs, updated_input_ids, nonoverlap_output_token_num, additional_tokens, additional_scores, temporary_collected_scores

def renew_pipeline(model_class):
    class JacobiPipeline(model_class):

        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

        def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
            cfg = self.cfg if hasattr(self, 'cfg') else cfg
            image_top_k = self.image_top_k if hasattr(self, 'image_top_k') else image_top_k
            text_top_k = self.text_top_k if hasattr(self, 'text_top_k') else text_top_k

            logits_processor = LogitsProcessorList()

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
            # past_attention_mask都是一样的,无需在意attention_mask
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
            all_collected_length = input_ids.shape[1]
            is_append_random_tokens = kwargs.get("is_append_random_tokens", False) ###!!!
            num_fill_rand_tokens = kwargs.get("num_fill_rand_tokens", 0)
            additional_tokens = kwargs.get("additional_tokens", None)
            additional_scores = kwargs.get("additional_scores", None)
            temporary_collected_scores = kwargs.get("temporary_collected_scores", None)

            generator = kwargs.get("generator", None)
            input_ids_accept_len_ptr = kwargs.get("input_ids_accept_len_ptr", 0)

            if is_append_random_tokens:
                
                if additional_tokens is not None:
                    kept_unconverged_token_num = num_fill_rand_tokens - additional_tokens.shape[-1]
                    rand_token_num = kept_unconverged_token_num if (kept_unconverged_token_num >= 0) else 0
                    additional_tokens_len = additional_tokens.shape[1]
                else:
                    kept_unconverged_token_num = 0
                    rand_token_num = num_fill_rand_tokens
                    additional_tokens_len = 0
                
                last_input_tokens = additional_tokens
                last_input_scores = additional_scores

                rand_tokens, scores_of_rand_tokens = get_multi_token_for_preparation(
                    img_vocab=self.img_vocab, rand_token_num=rand_token_num, 
                    input_ids=input_ids, temporary_collected_scores = temporary_collected_scores,
                    device=input_ids.device,
                    multi_token_init_scheme=self.multi_token_init_scheme,
                    last_input_tokens = last_input_tokens, last_input_scores = last_input_scores,
                    additional_tokens_len = additional_tokens_len,
                    img_width = kwargs.get("img_width", None),
                    generator = generator,
                    prefill_num = prefill_num,
                )

                if additional_tokens is not None:
                    input_ids = torch.cat([ 
                        input_ids, 
                        additional_tokens[:, : additional_tokens.shape[1] + kept_unconverged_token_num], 
                        rand_tokens,
                    ], dim=-1)
                else:
                    input_ids = torch.cat([ 
                        input_ids, 
                        rand_tokens,
                    ], dim=-1)

                input_token_scores = None
            
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            if past_key_values is not None:
                if isinstance(cache_position, torch.Tensor) and len(cache_position.shape) >= 2: ###!!!
                    if inputs_embeds is not None:  # Exception 1
                        input_ids = input_ids[cache_position]
                    elif input_ids.shape[1] != cache_position.shape[-1]:
                        input_ids = input_ids[cache_position]
                else:
                    if inputs_embeds is not None:  # Exception 1
                        input_ids = input_ids[:, -cache_position.shape[0] :]
                    elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                        input_ids = input_ids[:, cache_position]
                    
                    # if input_token_scores is not None:
                    #     input_token_scores = input_token_scores[:, cache_position]
                
                recycled_scores = additional_scores[
                    :, : additional_scores.shape[1] + kept_unconverged_token_num
                ] if additional_scores is not None else temporary_collected_scores[:, -1:-1]
                input_token_scores = gather_from_split_tensors(
                    tensor_list = [
                        temporary_collected_scores[:, -1:], 
                        recycled_scores, 
                        scores_of_rand_tokens,
                    ],
                    indexes = cache_position,
                    dim=1,
                    prefilled_length = all_collected_length - 1,
                    device=input_ids.device,
                )

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                if len(position_ids.shape) == 3: ###!!!
                    position_ids = position_ids[:, -1, :]
                
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases


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

            model_inputs['input_token_scores'] = input_token_scores
            model_inputs['input_ids_accept_len_ptr'] = input_ids_accept_len_ptr
            model_inputs['rand_tokens_shape'] = rand_tokens.shape[-1]

            return model_inputs

        def prepare_inputs_for_generation_jacobi_tree(
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
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            model_kwargs[cache_name] = cache
            if getattr(outputs, "state", None) is not None:
                model_kwargs["state"] = outputs.state

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
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
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )

            if model_kwargs.get("use_cache", True):
                # if num_new_tokens <= 1:
                #     model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
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

        def _update_model_kwargs_for_generation_tree(#需要修改为草稿树的样子
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
        
        def choose_tree_tokens(self, tree_input_ids_line, retrieve_indices, tree_input_scores_line):
            # 初始化存储最终结果的大列表
            big_indices_list = []
            input_ids_list = []
            input_scores_list = []
            final_indices = []
            final_tokens = []
            final_scores = []

            # Step1：将Tensor转到CPU并转为numpy数组（兼容GPU/CPU，且遍历高效）
            # 若Tensor是float类型（如-1.0），需先转int再处理，根据实际情况调整
            indices_np = retrieve_indices.cpu().numpy()
            input_ids_np = tree_input_ids_line.cpu().numpy()
            # Step2：转置（行→列，列→行），遍历转置后的行 = 遍历原Tensor的列
            indices_np = indices_np.transpose()
            input_ids_np = input_ids_np.transpose()
            
            # Step2：遍历每一行（与原逻辑一致，但基于numpy数组遍历）
            for row in indices_np:
                # 初始化当前行的唯一元素列表
                unique_row_list = []
                # 遍历当前行的每个元素
                for elem in row:
                    # 筛选：非-1 且 未在列表中（保证唯一）
                    if elem != -1 and elem not in unique_row_list:
                        unique_row_list.append(elem)
                # 将当前行结果加入大列表
                big_indices_list.append(unique_row_list)
            for i, row in enumerate(input_ids_np):
                # 初始化当前行的唯一元素列表
                unique_row_list = []
                unique_score_list = []
                # 遍历当前行的每个元素
                for j, elem in enumerate(row):
                    # 筛选：非-1 且 未在列表中（保证唯一）
                    if elem != -1 and elem != -2 and elem not in unique_row_list:
                        unique_row_list.append(elem)
                        unique_score_list.append(tree_input_scores_line[j][i])#行列转置了
                # 将当前行结果加入大列表
                input_ids_list.append(unique_row_list)
                input_scores_list.append(unique_score_list)
            # 遍历已有的token
            for i in range(len(input_ids_list)):
                for j in range(len(big_indices_list[i])):
                    if j >= len(input_ids_list[i]): #索引大,但是可用的token少了
                        continue
                    final_indices.append(big_indices_list[i][j])
                    final_tokens.append(input_ids_list[i][j])
                    final_scores.append(input_scores_list[i][j])
            return final_indices, final_tokens, final_scores
        
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
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:# init values
            pad_token_id = generation_config._pad_token_tensor
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
            max_num_new_tokens = self.max_num_new_tokens

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
                        (batch_size, ), dtype=dtype, device=device
                    ),
                )
            elif self.prefix_token_sampler_scheme == 'sjd++':
                prefix_token_sampler = SpeculativeSampler_PlusPlus(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                )
                prefix_token_sampler.threshold = generation_config.threshold
            elif self.prefix_token_sampler_scheme == 'max_coupling':
                prefix_token_sampler = SpeculativeSampler_MaxCoupling(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        (batch_size, ), dtype=dtype, device=device
                    ),
                )
            elif self.prefix_token_sampler_scheme == 'caccl':
                prefix_token_sampler = SpeculativeSampler_ContinuebyACCl(
                    generator=self.generator,
                    reject_sampling_relative_ids = -torch.ones(
                        batch_size, dtype=dtype, device=device,
                    ),
                    reject_sampling_draft_token_logits = torch.zeros(
                        (batch_size, self.config.vocab_size), dtype=dtype, device=device
                    ),
                    sampling_last_draft_token = torch.zeros(
                        (batch_size, ), dtype=dtype, device=device
                    ),
                )
                prefix_token_sampler.sequence_K = sequence_K
            # ensure only one sjd++ branch and consistent batch sizing
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
                # logging for diagnostics
                try:
                    logging.info(f"[sjd_pp][init] loop={gen_loop_num} cur_len={cur_len} matched={num_matched_tokens} threshold={getattr(prefix_token_sampler, 'threshold', None)}")
                except Exception:
                    pass

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
                model_kwargs = self._update_model_kwargs_for_generation_tree(
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
                # logging for diagnostics
                try:
                    logging.info(f"[sjd_pp][tree] loop={gen_loop_num} cur_len={cur_len} matched={num_matched_tokens} matched_retrieve_len={len(matched_retrieve_indices) if matched_retrieve_indices is not None else None} threshold={getattr(prefix_token_sampler, 'threshold', None)}")
                except Exception:
                    pass
                this_time_len = output_token_num
                output_token_num = max_num_new_tokens
                # accept_list.append(num_matched_tokens)# DEBUG

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
                # 核心步骤：生成非-1的掩码，过滤两个张量
                mask_retrieve_indices = retrieve_indices != -1
                tree_input_ids_line = tree_input_ids[:, retrieve_indices][0]
                tree_input_scores_line = tree_input_scores[:, retrieve_indices][0]
                tree_input_ids_line[mask_retrieve_indices==False] = -2
                rand_num = 0 #最长空缺序列
                for i in range(tree_input_ids_line.shape[0]):
                    rand_num = max(rand_num, (tree_input_ids_line[i] == -1).sum())
                #------------删掉等候区的无用的draft token------------
                if rand_num > 0:
                    tree_input_ids_line[:,1:] = torch.cat([tree_input_ids_line[:, 1:][:, rand_num:],tree_input_ids_line[:,1:][:, :rand_num]], dim=1)
                    tree_input_scores_line[:,1:] = torch.cat([tree_input_scores_line[:, 1:][:, rand_num:],tree_input_scores_line[:,1:][:, :rand_num]], dim=1)
                    tree_input_ids_line[:,1:][:, -rand_num:] = -1
                    final_indices_list, final_tokens_list, final_scores_list  = self.choose_tree_tokens(tree_input_ids_line, retrieve_indices, tree_input_scores_line)
                    tree_input_ids_restore = torch.full((1, tree_input_ids.shape[-1]), fill_value=-1,dtype=tree_input_ids.dtype,
                                                        device=tree_input_ids.device)          # [1, 9] 全 -1
                    tree_input_scores_restore = torch.zeros((1, tree_input_scores.shape[-2], tree_input_scores.shape[-1]),dtype=tree_input_scores.dtype,
                                                        device=tree_input_scores.device)
                    # 根据 final_indices_list 进行填充
                    for idx, token, score in zip(final_indices_list, final_tokens_list, final_scores_list):
                        tree_input_ids_restore[0, idx] = token                     # 填充 token
                        tree_input_scores_restore[0, idx] = score                  # 填充对应的 score vector
                else:
                    # flatten 恢复原状
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
                    # 核心步骤：生成非-1的掩码，过滤两个张量
                    mask_equal_invaild = flat_indices != -1
                    flat_indices = flat_indices[mask_equal_invaild]  # 过滤indices
                    flat_values = flat_values[0, mask_equal_invaild].unsqueeze(0)    # 同步过滤values
                    flat_values_scores = flat_values_scores[0, mask_equal_invaild].unsqueeze(0)
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
                # if rand_num > 0:# DEBUG
                #     flat_indices = retrieve_indices[:,1:].reshape(-1)       # [90]
                #     flat_values = prefix_token_sampler.zero_prob_tensor.reshape(1, -1)  # [1, 90]
                #     mask_equal_invaild = flat_indices != -1
                #     flat_indices = flat_indices[mask_equal_invaild]  # 过滤indices
                #     flat_values = flat_values[0, mask_equal_invaild].unsqueeze(0)    # 同步过滤values
                #     prob_list = torch.zeros(seq_len,dtype=prefix_token_sampler.zero_prob_tensor.dtype,device=tree_input_ids.device)
                #     prob_list.scatter_reduce_(0, flat_indices, flat_values[0], reduce='amax')
                #     prob_list=prob_list[1:] # 去掉root
                #     nonroot_index = self.buffers_tree_position_ids[1:]
                #     prob_all = layerwise_mean_simple(prob_list, nonroot_index)
                #     assert prob_all.shape[-1] != 0
                    # acrate_list.append(prob_all)# DEBUG

                # Store scores, attentions and hidden_states when required
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs
                if streamer is not None:
                    if len(next_tokens.shape) == 1:
                        streamer.put(next_tokens.cpu())
                    else:
                        for j in range(next_tokens.shape[1]):
                            streamer.put(next_tokens[:, j].cpu())
                
                # 更新attention mask, position ids, past key values
                model_kwargs = self._update_model_kwargs_for_generation_tree(
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

def renew_backbone(model_class):
    class JacobiBackbone(model_class):

        def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)

                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask
    
    return JacobiBackbone

def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)
    pipe_line.model._init_new_params(**kwargs)
    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)
    return pipe_line