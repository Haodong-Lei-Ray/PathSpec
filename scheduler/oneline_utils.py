import torch
import torch.nn.functional as F
import json
import torch
import numpy as np

class SpeculativeSampler_PlusPlus:

    def __init__(
        self, 
        collected_draft_logits=[], 
        collected_advanced_logits=[], 
        max_num_collected_logits=2,
        generator=None,
        draft_type = 'jacobian_states',
        reject_sampling_relative_ids = None,
        reject_sampling_draft_token_logits = None,
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

                if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
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
                            all_collected_input_ids[0, :],
                            resampled_target_tokens[b, :target_token_index],
                        ], dim=-1),
                    )
                    resampled_target_tokens[b, target_token_index] = resampled_tokens
                    # resampled_target_scores[b, target_token_index, :] = resampled_scores # the score is kept, so not to update this.
                    first_misaligned_token_index = i

                    break

            threshold = self.threshold
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(x^{j}|x^{j-1}_:)
                C = sampled_advanced_prob / (sampled_draft_prob + 1e-7)
                if C > threshold:
                    resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores


class SpeculativeSampler_MaxCoupling:

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

                if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
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

            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j-1}_:
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(.|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(.|x^{j-1}_:)

                r = rs[b, i, cls_idx]

                if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:
                    resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                            token_advanced_prob = advanced_prob[b, target_token_index, :], # p(.|x^{j}_:)
                            token_draft_prob = draft_prob[b, draft_token_index, :], # p(.|x^{j-1}_:)
                            logits_processor = logits_processor,
                            logits_warper = logits_warper,
                            all_collected_input_ids = torch.cat([
                                all_collected_input_ids[b, :],
                                resampled_target_tokens[b, :target_token_index],
                            ], dim=-1),
                        )
                    resampled_target_tokens[b, target_token_index] = resampled_tokens

            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSampler_ContinuebyACCl:

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
    
    def canculate_C1(self, accept_list):
        C1 = 1 if accept_list[0] else 0
        for i in range(1, len(accept_list)):
            if accept_list[i]:
                C1+=1
            else:
                return C1
        return C1
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
            index_list = torch.arange(1, L)
            draft_token_index_list = draft_token_index_selector(index_list) # draft token 的索引
            target_token_index_list = advanced_token_index_selector(index_list) # 待验证的索引, 也就是此刻采样的概率
            cls_idx_list = draft_tokens[b, draft_token_index_list] #draft token的index
            sampled_advanced_prob_list = advanced_prob[b, target_token_index_list, cls_idx_list]
            sampled_draft_prob_list = draft_prob[b, draft_token_index_list, cls_idx_list]
            accp_list = (sampled_advanced_prob_list / sampled_draft_prob_list).clamp(max=1)
            r_list = rs[b, target_token_index_list, cls_idx_list]
            accept_list = r_list < accp_list
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                # draft_token_index = draft_token_index_selector(i) # draft token 的索引
                draft_token_index = draft_token_index_list[i-1]
                # target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率
                target_token_index = target_token_index_list[i-1]

                # cls_idx = draft_tokens[b, draft_token_index] #draft token的index
                cls_idx = cls_idx_list[i-1]

                # sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]
                sampled_advanced_prob = sampled_advanced_prob_list[i-1]

                # sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                sampled_draft_prob = sampled_draft_prob_list[i-1]
                # r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
                if accept_list[i-1]: 
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

            threshold = self.threshold
            threshold1 = 2
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(x^{j}|x^{j-1}_:)
                C = sampled_advanced_prob / (sampled_draft_prob + 1e-7)
                if C > threshold:
                    resampled_target_tokens[b, target_token_index] = cls_idx
                else:
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1:
                        resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSampler_ContinuebyACCl_PlusPlus:

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
    
    def canculate_C1(self, accept_list):
        C1 = 1 if accept_list[0] else 0
        for i in range(1, len(accept_list)):
            if accept_list[i]:
                C1+=1
            else:
                return C1
        return C1
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
            index_list = torch.arange(1, L)
            draft_token_index_list = draft_token_index_selector(index_list) # draft token 的索引
            target_token_index_list = advanced_token_index_selector(index_list) # 待验证的索引, 也就是此刻采样的概率
            cls_idx_list = draft_tokens[b, draft_token_index_list] #draft token的index
            sampled_advanced_prob_list = advanced_prob[b, target_token_index_list, cls_idx_list]
            sampled_draft_prob_list = draft_prob[b, draft_token_index_list, cls_idx_list]
            accp_list = (sampled_advanced_prob_list / sampled_draft_prob_list).clamp(max=1)
            r_list = rs[b, target_token_index_list, cls_idx_list]
            accept_list = r_list < accp_list
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                # draft_token_index = draft_token_index_selector(i) # draft token 的索引
                draft_token_index = draft_token_index_list[i-1]
                # target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率
                target_token_index = target_token_index_list[i-1]

                # cls_idx = draft_tokens[b, draft_token_index] #draft token的index
                cls_idx = cls_idx_list[i-1]

                # sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]
                sampled_advanced_prob = sampled_advanced_prob_list[i-1]

                # sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                sampled_draft_prob = sampled_draft_prob_list[i-1]
                # r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
                if accept_list[i-1]: 
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

            # threshold = 0.5
            threshold1 = 2
            threshold2 = 6
            # check_list = []
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                # sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                advanced_prob_list = advanced_prob[b, target_token_index]
                advanced_prob_entropy = -torch.sum(advanced_prob_list * torch.log(advanced_prob_list + torch.finfo(advanced_prob_list.dtype).eps))
                C1 = self.canculate_C1(accept_list[i-1:])
                
                if C1 > threshold1 and advanced_prob_entropy > threshold2:
                    resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
            # from scipy import stats  # 众数需scipy（你的环境支持）
            # arr = np.array(check_list)
            # # 计算统计量
            # mean_val = np.mean(arr)           # 平均值
            # median_val = np.median(arr)       # 中位数
            # mode_val = stats.mode(arr, keepdims=True).mode[0]  # 众数（返回最频繁值）
            # print(f"平均值 (Mean): {mean_val}")
            # print(f"中位数 (Median): {median_val}")
            # print(f"众数 (Mode): {mode_val}")
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSampler_RelaxedbyACCl(SpeculativeSampler_ContinuebyACCl):

    def canculate_C1(self, accept_list):
        C1 = 1 if accept_list[0] else 0
        for i in range(1, len(accept_list)):
            if accept_list[i]:
                C1+=1
            else:
                return C1
        return C1
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
            index_list = torch.arange(1, L)
            draft_token_index_list = draft_token_index_selector(index_list) # draft token 的索引
            target_token_index_list = advanced_token_index_selector(index_list) # 待验证的索引, 也就是此刻采样的概率
            cls_idx_list = draft_tokens[b, draft_token_index_list] #draft token的index
            sampled_advanced_prob_list = advanced_prob[b, target_token_index_list, cls_idx_list]
            sampled_draft_prob_list = draft_prob[b, draft_token_index_list, cls_idx_list]
            accp_list = (sampled_advanced_prob_list / sampled_draft_prob_list).clamp(max=1)
            r_list = rs[b, target_token_index_list, cls_idx_list]
            accept_list = r_list < accp_list
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                draft_token_index = draft_token_index_list[i-1]
                target_token_index = target_token_index_list[i-1]

                cls_idx = cls_idx_list[i-1]

                sampled_advanced_prob = sampled_advanced_prob_list[i-1]

                sampled_draft_prob = sampled_draft_prob_list[i-1]

                self.sampling_last_draft_token[b] = cls_idx

                threshold1 = 2
                if accept_list[i-1]: 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1:
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

            threshold = self.threshold
            threshold1 = 2
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(x^{j}|x^{j-1}_:)
                C = sampled_advanced_prob / (sampled_draft_prob + 1e-7)
                if C > threshold:
                    resampled_target_tokens[b, target_token_index] = cls_idx
                else:
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1:
                        resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSampler_RelaxedbyACCl_Entropy(SpeculativeSampler_ContinuebyACCl):

    def canculate_C1(self, accept_list):
        C1 = 1 if accept_list[0] else 0
        for i in range(1, len(accept_list)):
            if accept_list[i]:
                C1+=1
            else:
                return C1
        return C1
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
            index_list = torch.arange(1, L)
            draft_token_index_list = draft_token_index_selector(index_list) # draft token 的索引
            target_token_index_list = advanced_token_index_selector(index_list) # 待验证的索引, 也就是此刻采样的概率
            cls_idx_list = draft_tokens[b, draft_token_index_list] #draft token的index
            advanced_prob_list = advanced_prob[b, target_token_index_list]
            sampled_advanced_prob_list = advanced_prob[b, target_token_index_list, cls_idx_list]
            sampled_draft_prob_list = draft_prob[b, draft_token_index_list, cls_idx_list]
            accp_list = (sampled_advanced_prob_list / sampled_draft_prob_list).clamp(max=1)
            r_list = rs[b, target_token_index_list, cls_idx_list]
            accept_list = r_list < accp_list
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                draft_token_index = draft_token_index_list[i-1]
                target_token_index = target_token_index_list[i-1]

                cls_idx = cls_idx_list[i-1]

                sampled_advanced_prob = sampled_advanced_prob_list[i-1]

                sampled_draft_prob = sampled_draft_prob_list[i-1]
                advanced_prob_list = advanced_prob_list[i-1]
                advanced_prob_entropy = -torch.sum(advanced_prob_list * torch.log(advanced_prob_list + torch.finfo(advanced_prob_list.dtype).eps))

                self.sampling_last_draft_token[b] = cls_idx

                threshold1 = 2
                threshold2 = 16
                if accept_list[i-1]: 
                    # accept sampling
                    resampled_target_tokens[b, target_token_index] = cls_idx
                    resampled_target_scores[b, target_token_index, :] = draft_prob[b, draft_token_index, :]
                else:
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1 and advanced_prob_entropy > threshold2:
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

            threshold = self.threshold
            threshold1 = 1
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(x^{j}|x^{j-1}_:)
                C = sampled_advanced_prob / (sampled_draft_prob + 1e-7)
                if C > threshold:
                    resampled_target_tokens[b, target_token_index] = cls_idx
                else:
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1:
                        resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSampler_ContinuebyACCl_Entropy(SpeculativeSampler_ContinuebyACCl):

    def canculate_C1(self, accept_list):
        C1 = 1 if accept_list[0] else 0
        for i in range(1, len(accept_list)):
            if accept_list[i]:
                C1+=1
            else:
                return C1
        return C1
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
            index_list = torch.arange(1, L)
            draft_token_index_list = draft_token_index_selector(index_list) # draft token 的索引
            target_token_index_list = advanced_token_index_selector(index_list) # 待验证的索引, 也就是此刻采样的概率
            cls_idx_list = draft_tokens[b, draft_token_index_list] #draft token的index
            sampled_advanced_prob_list = advanced_prob[b, target_token_index_list, cls_idx_list]
            sampled_draft_prob_list = draft_prob[b, draft_token_index_list, cls_idx_list]
            accp_list = (sampled_advanced_prob_list / sampled_draft_prob_list).clamp(max=1)
            r_list = rs[b, target_token_index_list, cls_idx_list]
            accept_list = r_list < accp_list
            first_misaligned_token_index = L # keep at least one token left
            for i in range(1, L):

                # draft_token_index = draft_token_index_selector(i) # draft token 的索引
                draft_token_index = draft_token_index_list[i-1]
                # target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率
                target_token_index = target_token_index_list[i-1]

                # cls_idx = draft_tokens[b, draft_token_index] #draft token的index
                cls_idx = cls_idx_list[i-1]

                # sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]
                sampled_advanced_prob = sampled_advanced_prob_list[i-1]

                # sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                sampled_draft_prob = sampled_draft_prob_list[i-1]
                # r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
                if accept_list[i-1]: 
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

            threshold = self.threshold
            threshold1 = 1
            threshold2 = 5
            for i in range(first_misaligned_token_index+1, L):
                draft_token_index = draft_token_index_selector(i) 
                target_token_index = advanced_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index] #x^{j}
                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx] # p(x^{j}|x^{j}_:)
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx] # p(x^{j}|x^{j-1}_:)
                C = sampled_advanced_prob / (sampled_draft_prob + 1e-7)
                if C > threshold:
                    resampled_target_tokens[b, target_token_index] = cls_idx
                else:
                    advanced_prob_list = advanced_prob[b, target_token_index]
                    advanced_prob_entropy = -torch.sum(advanced_prob_list * torch.log(advanced_prob_list + torch.finfo(advanced_prob_list.dtype).eps))
                    C1 = self.canculate_C1(accept_list[i-1:])
                    if C1 > threshold1 and advanced_prob_entropy > threshold2:
                        resampled_target_tokens[b, target_token_index] = cls_idx
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores


def get_multi_token_for_preparation(
    img_vocab, 
    rand_token_num, 
    input_ids, temporary_collected_scores, device, 
    multi_token_init_scheme=None,
    last_input_tokens=None, last_input_scores=None,
    generator = None,
    eps = 1e-7,
    prefill_num = 3,
    additional_tokens_len = 0,
    **kwargs,
):

    def random_multinomial_sample_from_logits(rand_logits):
        logits = rand_logits
        probs_shape = None
        if len(logits.shape) >= 3:
            probs_shape = logits.shape
            logits = logits.flatten(0, len(probs_shape)-2)

        topk_logits, topk_cls_indices = torch.topk(logits, k=1, dim=-1)
        logits = torch.full_like(logits, -float("Inf")).scatter(-1, topk_cls_indices, topk_logits)
        probs = F.softmax(logits, dim=-1)

        rand_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if probs_shape is not None:
            rand_tokens = rand_tokens.reshape(probs_shape[:-1])
            probs = probs.reshape(probs_shape)
        
        return rand_tokens, probs

    if multi_token_init_scheme == 'random':
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab)
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens]

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)
    
    else:
        img_vocab = img_vocab.to(device)
        img_vocab_size = len(img_vocab)
        rand_tokens = torch.randint(
            0, img_vocab_size, 
            (*input_ids.shape[:-1], rand_token_num)
        ).to(device)
        rand_tokens = img_vocab[rand_tokens]

        scores_of_rand_tokens = temporary_collected_scores.new_zeros(
            (*temporary_collected_scores.shape[:-2], rand_token_num, temporary_collected_scores.shape[-1])
        )
        scores_of_rand_tokens = torch.scatter(scores_of_rand_tokens, -1, rand_tokens.unsqueeze(-1), 1.0)


        img_width = kwargs.get("img_width", None)
        pad_len = 1
        img_width = img_width + pad_len if img_width is not None else 0
        input_ids_len = input_ids.shape[1]
        
        prefill_num = prefill_num + 3 # TODO: this is for mgpt, the first 3 tokens <start, h, w>
        if (img_width > 0) and (input_ids_len + additional_tokens_len >= prefill_num) and (rand_token_num > 0):
            
            horizon_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) % img_width
            vertical_indices = (torch.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                device=device, dtype=torch.long
            ) - prefill_num) // img_width

            if 'horizon' in multi_token_init_scheme:
                valid_indices = (horizon_indices - 1 >= 0)
                last_vertical_indices = vertical_indices
                last_horizon_indices = horizon_indices - 1
            else:
                # # vertical consumes more memory to store the previous logits, so we use horizon in practice
                # if 'vertical' in multi_token_init_scheme:
                #     valid_indices = (vertical_indices - 1 >= 0)
                #     last_vertical_indices = vertical_indices - 1
                #     last_horizon_indices = horizon_indices
                assert False, f"multi_token_init_scheme should be 'horizon' or 'vertical', but got {multi_token_init_scheme}"
            
            last_input_tokens = torch.cat(
                [input_ids, last_input_tokens], dim=1
            ) if last_input_tokens is not None else input_ids
            last_input_scores = torch.cat(
                [temporary_collected_scores, last_input_scores], dim=1
            ) if last_input_scores is not None else temporary_collected_scores
            last_input_logits = (last_input_scores.float() + eps).log()

            last_flatten_indices = last_vertical_indices[valid_indices] * img_width + last_horizon_indices[valid_indices] + prefill_num
            
            # e.g., last indices [100, 101, 102], but the current indices up to 100, 
            # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
            last_flatten_indices = last_flatten_indices.clamp(min=0, max = last_input_tokens.shape[1]-1) 
            
            last_resampled_input_tokens = last_input_tokens[:, last_flatten_indices]
            # last_resampled_input_logits = last_input_logits[:, last_flatten_indices]

            if 'sample' in multi_token_init_scheme:
                resampled_rand_tokens, resampled_scores_of_rand_tokens = random_multinomial_sample_from_logits(
                    last_resampled_input_logits
                )
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            elif 'repeat' in multi_token_init_scheme:
                resampled_rand_tokens = last_resampled_input_tokens
                scores_of_rand_tokens[:, valid_indices] = 0
                scores_of_rand_tokens[:, valid_indices] = torch.scatter(
                    scores_of_rand_tokens[:, valid_indices], -1, resampled_rand_tokens.unsqueeze(-1), 1.0)
            else:
                assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
            
            rand_tokens[:, valid_indices] = resampled_rand_tokens
    
    return rand_tokens, scores_of_rand_tokens