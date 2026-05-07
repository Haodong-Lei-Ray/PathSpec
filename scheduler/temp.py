
def renew_sampler(model_class):
    
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
            max_length = generation_config.max_length
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample
            # if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            #     raise ValueError(
            #         "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            #         f"{logits_warper})."
            #     )

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
            elif self.prefix_token_sampler_scheme == 'jacobi':
                prefix_token_sampler = None
            else:
                raise ValueError(f"prefix_token_sampler_scheme: {self.prefix_token_sampler_scheme}")

            count_time = True
            if count_time:
                t1 = torch.cuda.Event(enable_timing=True)
                t2 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t1.record()

            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
            ):
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

                # Store scores, attentions and hidden_states when required
                assert (not return_dict_in_generate) # TODO: too many codes to collect the prefixes in outputs
                # if return_dict_in_generate:
                #     if output_scores:
                #         scores += (next_token_scores,) if len(next_token_scores.shape) == 3 else tuple(next_token_scores.unbind(dim=-2))
                #     if output_logits:
                #         raw_logits += (next_token_logits,) if len(next_token_logits.shape) == 3 else tuple(next_token_logits.unbind(dim=-2))
                #     if output_attentions:
                #         decoder_attentions += (
                #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                #         )
                #         if self.config.is_encoder_decoder:
                #             cross_attentions += (outputs.cross_attentions,)

                #     if output_hidden_states:
                #         decoder_hidden_states += (
                #             (outputs.decoder_hidden_states,)
                #             if self.config.is_encoder_decoder
                #             else (outputs.hidden_states,)
                #         )

                if streamer is not None:
                    if len(next_tokens.shape) == 1:
                        streamer.put(next_tokens.cpu())
                    else:
                        for j in range(next_tokens.shape[1]):
                            streamer.put(next_tokens[:, j].cpu())
                
                # No input_ids here, only the cache_position
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    num_new_tokens = nonoverlap_output_token_num,
                )

                # check whether we get the end token
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
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
