import torch
import torch.nn.functional as F
import numpy as np

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

        self._init_reject_sampling_params(B=1)
    
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
        
    def _init_reject_sampling_params(self, B):
        self.reject_sampling_draft_token_logits = torch.zeros(
            (B, self.reject_sampling_draft_token_logits.shape[-1]), dtype=self.reject_sampling_draft_token_logits.dtype, device=self.reject_sampling_draft_token_logits.device
        )
        self.reject_sampling_relative_ids = -torch.ones(
            B, dtype=self.reject_sampling_relative_ids.dtype, device=self.reject_sampling_relative_ids.device,
        )
        self.sampling_last_draft_token = torch.zeros(
            B, dtype=self.sampling_last_draft_token[0].dtype, device=self.sampling_last_draft_token[0].device
        )
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        retrieve_indices = kwargs.get("retrieve_indices", None)
        if retrieve_indices is not None:
            stop_retrieve_indices = (retrieve_indices >= 0).sum(dim=1)
        B, L = draft_tokens.shape

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params(B)

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        # self.zero_prob_tensor = torch.full_like(draft_tokens[...,1:], fill_value=-1.0).to(torch.float)#DEBUG
        
        for b in range(B):
            if retrieve_indices is not None:
                first_misaligned_token_index = stop_retrieve_indices[b].item() # keep at least one token left
                node_len = stop_retrieve_indices[b].item() 
            else:
                first_misaligned_token_index = L
                node_len =  L
            # for i in range(1, L):
            for i in range(1, node_len):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx
                
                if r < (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1): 
                    # accept sampling
                    # self.zero_prob_tensor[b, target_token_index] = (sampled_advanced_prob  / sampled_draft_prob).clamp(max=1)#DEBUG
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
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeGroupSumSampler:

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
        groupsum_delta=0.01
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

        self._init_reject_sampling_params(B=1)
        # MSCR
        self.groupsum_delta = groupsum_delta
    
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
        
    def _init_reject_sampling_params(self, B):
        self.reject_sampling_draft_token_logits = torch.zeros(
            (B, self.reject_sampling_draft_token_logits.shape[-1]), dtype=self.reject_sampling_draft_token_logits.dtype, device=self.reject_sampling_draft_token_logits.device
        )
        self.reject_sampling_relative_ids = -torch.ones(
            B, dtype=self.reject_sampling_relative_ids.dtype, device=self.reject_sampling_relative_ids.device,
        )
        self.sampling_last_draft_token = torch.zeros(
            B, dtype=self.sampling_last_draft_token[0].dtype, device=self.sampling_last_draft_token[0].device
        )
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        retrieve_indices = kwargs.get("retrieve_indices", None)
        if retrieve_indices is not None:
            stop_retrieve_indices = (retrieve_indices >= 0).sum(dim=1)
        B, L = draft_tokens.shape

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params(B)

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        for b in range(B):
            if retrieve_indices is not None:
                first_misaligned_token_index = stop_retrieve_indices[b].item() # keep at least one token left
                node_len = stop_retrieve_indices[b].item() 
            else:
                first_misaligned_token_index = L
                node_len =  L
            # for i in range(1, L):
            for i in range(1, node_len):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index
                cross_cls_idx = torch.cat((draft_tokens[:b, draft_token_index], draft_tokens[b+1:, draft_token_index]), dim = 0)

                self_advanced_prob = advanced_prob[b, target_token_index, cls_idx]
                
                cross_advanced_prob = advanced_prob[b, target_token_index, cross_cls_idx]
                cross_advanced_prob = cross_advanced_prob.sum()

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                neigbhor_thr = self.groupsum_delta
                assert self.sequence_K > 1
                sampled_advanced_prob = self_advanced_prob + cross_advanced_prob * neigbhor_thr / (self.sequence_K-1)
                acp = (sampled_advanced_prob  / sampled_draft_prob).sum(dim=0)
                if r < acp.clamp(max=1): 
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
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class SpeculativeSamplerLantern:

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
        lantern_delta = 3,
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

        self._init_reject_sampling_params(B=1)

        # Lantern 相关变量
        self.lantern_k = 10
        self.lantern_delta = lantern_delta
        self.image_token_offset = 4 # 前四个token是特殊token，不参与lantern计算
        
        while 1: 
            try:
                nearest_latents_path = hf_hub_download("sihwanpark/LANTERN-Lumina-mGPT-7B-768", "top_8191_indices.npy")
                self.nearest_latents = np.load(nearest_latents_path)
                break
            except:
                print(f"Network wrong in top_8191_indices.npy")
                pass
    
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
        
    def _init_reject_sampling_params(self, B):
        self.reject_sampling_draft_token_logits = torch.zeros(
            (B, self.reject_sampling_draft_token_logits.shape[-1]), dtype=self.reject_sampling_draft_token_logits.dtype, device=self.reject_sampling_draft_token_logits.device
        )
        self.reject_sampling_relative_ids = -torch.ones(
            B, dtype=self.reject_sampling_relative_ids.dtype, device=self.reject_sampling_relative_ids.device,
        )
        self.sampling_last_draft_token = torch.zeros(
            B, dtype=self.sampling_last_draft_token[0].dtype, device=self.sampling_last_draft_token[0].device
        )
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        retrieve_indices = kwargs.get("retrieve_indices", None)
        if retrieve_indices is not None:
            stop_retrieve_indices = (retrieve_indices >= 0).sum(dim=1)
        B, L = draft_tokens.shape

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params(B)

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        for b in range(B):
            if retrieve_indices is not None:
                first_misaligned_token_index = stop_retrieve_indices[b].item() # keep at least one token left
                node_len = stop_retrieve_indices[b].item() 
            else:
                first_misaligned_token_index = L
                node_len =  L
            # for i in range(1, L):
            for i in range(1, node_len):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # -------lantern-------
                px = sampled_advanced_prob
                qx = sampled_draft_prob
                
                # Lantern 调整 px（从 anole 适配）
                if cls_idx - self.image_token_offset >= self.nearest_latents.shape[0]:
                    nearest_probs = torch.ones(self.lantern_k, device=advanced_prob.device)
                else:
                    nearest_probs = advanced_prob[b, target_token_index, self.nearest_latents[cls_idx - self.image_token_offset, :self.lantern_k] + self.image_token_offset]
                cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=0)
                if self.lantern_delta > 1.0:
                    indices = (cumsum_nearest_probs <= (self.lantern_delta - 1) * px).nonzero(as_tuple=True)[0]
                else:
                    indices = (cumsum_nearest_probs <= self.lantern_delta).nonzero(as_tuple=True)[0]

                if indices.numel() == 0:
                    indices = -1
                else:
                    indices = indices[-1].item()  # 取最后一个有效索引，并转为 scalar

                if indices == -1:
                    px = px
                else:
                    px = px + cumsum_nearest_probs[indices]

                assert qx >= 0
                acp = (px / qx).clamp(max=1)
                # -------lantern end-------
                if r < acp: 
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
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class GroupSpeculativeSampler:

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

        self._init_reject_sampling_params(B=1)

        #GSD
        self.head_img_sims = None
    
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
        
    def _init_reject_sampling_params(self, B):
        self.reject_sampling_draft_token_logits = torch.zeros(
            (B, self.reject_sampling_draft_token_logits.shape[-1]), dtype=self.reject_sampling_draft_token_logits.dtype, device=self.reject_sampling_draft_token_logits.device
        )
        self.reject_sampling_relative_ids = -torch.ones(
            B, dtype=self.reject_sampling_relative_ids.dtype, device=self.reject_sampling_relative_ids.device,
        )
        self.sampling_last_draft_token = torch.zeros(
            B, dtype=self.sampling_last_draft_token[0].dtype, device=self.sampling_last_draft_token[0].device
        )
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        retrieve_indices = kwargs.get("retrieve_indices", None)
        if retrieve_indices is not None:
            stop_retrieve_indices = (retrieve_indices >= 0).sum(dim=1)
        B, L = draft_tokens.shape

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params(B)

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        
        logit_sort = advanced_prob.sort(dim=-1).indices
        
        for b in range(B):
            if retrieve_indices is not None:
                first_misaligned_token_index = stop_retrieve_indices[b].item() # keep at least one token left
                node_len = stop_retrieve_indices[b].item() 
            else:
                first_misaligned_token_index = L
                node_len =  L
            # for i in range(1, L):
            for i in range(1, node_len):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # -------GSD-------
                group_size = 7
                cur_idx = torch.nonzero(logit_sort[b,target_token_index]==cls_idx)
                logit_sort_idx = logit_sort[b,target_token_index][cur_idx-group_size:cur_idx+group_size] #logit-based
                try :
                    adv_super_prob      = advanced_prob[b,target_token_index][logit_sort_idx]
                    draft_super_prob    = draft_prob[b,draft_token_index][logit_sort_idx]

                    #"""Prob filtering"""
                    p_thr = 0.15
                    d_thr = 0.5

                    is_in_p_thr =  ( (adv_super_prob - advanced_prob[b,target_token_index][cls_idx]).abs() < p_thr ).float()
                    if logit_sort_idx.max().item() <= 10000 :
                        is_in_d_thr = ( self.head_img_sims[cls_idx][logit_sort_idx] < d_thr ).float()
                    else :
                        is_in_d_thr = 1.

                    final_mask = (is_in_p_thr * is_in_d_thr).float()

                    #print(final_mask)

                    adv_super_prob      = adv_super_prob*final_mask
                    draft_super_prob    = draft_super_prob*final_mask
                    #"""Prob filtering"""

                    adv_super_prob      = adv_super_prob.sum()
                    draft_super_prob    = draft_super_prob.sum()
                    final_p = (adv_super_prob/ draft_super_prob).clamp(max=1)
                except Exception as e :
                    final_p = (sampled_advanced_prob / sampled_draft_prob).clamp(max=1)

                global saver
                # -------GSD end-------
                if r < final_p: 
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
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class GroupSpeculativeGroupSumSampler:

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
        groupsum_delta = 0.01
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

        self._init_reject_sampling_params(B=1)

        #GSD
        self.head_img_sims = None
        self.groupsum_delta = groupsum_delta
    
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
        
    def _init_reject_sampling_params(self, B):
        self.reject_sampling_draft_token_logits = torch.zeros(
            (B, self.reject_sampling_draft_token_logits.shape[-1]), dtype=self.reject_sampling_draft_token_logits.dtype, device=self.reject_sampling_draft_token_logits.device
        )
        self.reject_sampling_relative_ids = -torch.ones(
            B, dtype=self.reject_sampling_relative_ids.dtype, device=self.reject_sampling_relative_ids.device,
        )
        self.sampling_last_draft_token = torch.zeros(
            B, dtype=self.sampling_last_draft_token[0].dtype, device=self.sampling_last_draft_token[0].device
        )
    
    def __call__(
        self, draft_tokens, advanced_tokens, draft_prob, advanced_prob,
        logits_processor = None, logits_warper = None,
        all_collected_input_ids = None,
        **kwargs,
    ): 
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        retrieve_indices = kwargs.get("retrieve_indices", None)
        if retrieve_indices is not None:
            stop_retrieve_indices = (retrieve_indices >= 0).sum(dim=1)
        B, L = draft_tokens.shape

        # reinitalize self.reject_sampling_relative_ids
        self._init_reject_sampling_params(B)

        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device, generator=self.generator)

        draft_token_index_selector = self.draft_token_index_selector
        advanced_token_index_selector = self.advanced_token_index_selector

        resampled_target_tokens = advanced_tokens.clone()
        resampled_target_scores = advanced_prob.clone()

        first_misaligned_token_inds = []
        
        logit_sort = advanced_prob.sort(dim=-1).indices
        
        for b in range(B):
            if retrieve_indices is not None:
                first_misaligned_token_index = stop_retrieve_indices[b].item() # keep at least one token left
                node_len = stop_retrieve_indices[b].item() 
            else:
                first_misaligned_token_index = L
                node_len =  L
            # for i in range(1, L):
            for i in range(1, node_len):

                draft_token_index = draft_token_index_selector(i) # draft token 的索引
                target_token_index = advanced_token_index_selector(i) # 待验证的索引, 也就是此刻采样的概率

                cls_idx = draft_tokens[b, draft_token_index] #draft token的index

                sampled_advanced_prob = advanced_prob[b, target_token_index, cls_idx]

                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]

                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token[b] = cls_idx

                # -------GSD-------
                group_size = 7
                cur_idx = torch.nonzero(logit_sort[b,target_token_index]==cls_idx)
                logit_sort_idx = logit_sort[b,target_token_index][cur_idx-group_size:cur_idx+group_size] #logit-based
                target_cur_idx = torch.nonzero(logit_sort[:,target_token_index]==draft_tokens[:, draft_token_index, None])[:, 1:]
                cross_logit_sort_idx = None
                for bi in range(B):
                    if bi != b :
                        if cross_logit_sort_idx is None:
                            cross_logit_sort_idx = logit_sort[bi,target_token_index][target_cur_idx[bi]-group_size:target_cur_idx[bi]+group_size] #logit-based
                        else:
                            cross_logit_sort_idx = torch.unique(torch.cat([cross_logit_sort_idx, logit_sort[bi,target_token_index][target_cur_idx[bi]-group_size:target_cur_idx[bi]+group_size]], dim=0))

                try :

                    #larger
                    adv_super_prob      = advanced_prob[b,target_token_index][logit_sort_idx]
                    cross_adv_super_prob      = advanced_prob[b,target_token_index][cross_logit_sort_idx]
                    draft_super_prob    = draft_prob[b,draft_token_index][logit_sort_idx]

                    #"""Neighbor filtering"""
                    neigbhor_thr = self.groupsum_delta

                    #"""Prob filtering"""
                    p_thr = 0.15
                    d_thr = 0.5

                    is_in_p_thr =  ( (adv_super_prob - advanced_prob[b,target_token_index][cls_idx]).abs() < p_thr ).float()
                    if logit_sort_idx.max().item() <= 10000 :
                        is_in_d_thr = ( self.head_img_sims[cls_idx][logit_sort_idx] < d_thr ).float()
                    else :
                        is_in_d_thr = 1.
                    final_mask = (is_in_p_thr * is_in_d_thr).float()
                    
                    is_in_p_thr =  ( (cross_adv_super_prob - advanced_prob[b,target_token_index][cls_idx]).abs() < p_thr ).float()
                    if cross_logit_sort_idx.max().item() <= 10000 :
                        is_in_d_thr = ( self.head_img_sims[cls_idx][cross_logit_sort_idx] < d_thr ).float()
                    else :
                        is_in_d_thr = 1.
                    cross_final_mask = (is_in_p_thr * is_in_d_thr).float()

                    # original end
                    cross_adv_super_prob = cross_adv_super_prob*cross_final_mask

                    adv_super_prob = adv_super_prob*final_mask
                    draft_super_prob = draft_super_prob*final_mask

                    assert self.sequence_K > 1
                    cross_adv_super_prob = cross_adv_super_prob.sum() * neigbhor_thr/(self.sequence_K - 1)
                    self_adv_super_prob = adv_super_prob.sum()
                    draft_super_prob    = draft_super_prob.sum()

                    findal_adv_super_prob = self_adv_super_prob + cross_adv_super_prob
                    final_p_larger = (findal_adv_super_prob/ draft_super_prob).clamp(max=1)

                    final_p = final_p_larger
                except Exception as e :
                    final_p = (sampled_advanced_prob / sampled_draft_prob).clamp(max=1)

                global saver
                # -------GSD end-------
                if r < final_p: 
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
        
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_scores

class Result:
    def __init__(self, input_ids, loop_num, token_gen_len, time_forward):
        self.input_ids=input_ids
        self.loop_num=loop_num
        self.token_gen_len=token_gen_len
        self.time_forward=time_forward
