import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
import gc

import os
import time

from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch

from emu3.mllm.processing_emu3 import Emu3Processor

import argparse

import random
import numpy as np

import json, csv
import re

import lumina_mgpt.data.drafters.choices as choices
def load_prompts(args):
    prompts = []
    output_file_name_list = []
    if args.prompt == "PartiPrompts":
        with open('data/prompts/PartiPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            ids = 0
            for row in tsv_reader:
                prompts.append(row['Prompt'])
                output_file_name_list.append(ids)
                ids += 1
    elif args.prompt == "MSCOCO2017Val":#
        # with open('data/prompts/captions_val2017_longest.json', 'r') as f:
        #     captions = json.load(f)
        #     for caption in captions:
        #         prompts.append(caption)
        from pycocotools.coco import COCO
        coco = COCO("data/prompts/captions_val2017.json")
        top_k = 0
        for i in range(args.num_images):
            img_id = coco.getImgIds()[i]
            img_name = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for j, ann in enumerate(anns):
                ann_id = ann['id']
                caption = ann["caption"]
                prompts.append(caption)
                output_file_name_list.append(ann_id)
                if j == top_k:
                    break
    elif args.prompt == "MSCOCO2014Val":
        with open('data/prompts/captions_val_2014.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "MSCOCO2017Train":
        with open('data/prompts/captions_train2017_extracted.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption['caption'])
    elif args.prompt == "SJDPrompts":
        with open('data/prompts/SJDPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    elif args.prompt == "T2ICompBenchVal":
        with open("data/prompts/T2I-CompBench_val.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for line in data:
            # 每行是一个独立的JSON对象，逐行解析
            prompts.append(line['caption'])
            output_file_name_list.append(line['image_id'])
    else:
        # Single prompt input
        prompts = [args.prompt] * args.num_images
        output_file_name_list = [i for i in range(len(prompts))]

    if args.slice is not None:
        assert re.match(r'^\d+-\d+$', args.slice), f"Invalid format: '{args.slice}'. Expected format is 'start-end'."

        start, end = map(int, args.slice.split('-'))
        assert start < end, f"Invalid range: '{args.slice}'. Start value must be less than end value."
        assert start >= 0 and end >= 0, "Slice values must be non-negative."

        prompts = prompts[start:end]
        output_file_name_list = output_file_name_list[start:end]
    
    if args.num_images < len(prompts):
        print(f"Number of images to generate is less than the number of prompts. Sampling {args.num_images} prompts.")
        if args.benchmark_way == "random":
            prompts = random.sample(prompts, args.num_images)
        else:
            prompts = prompts[:args.num_images]
            output_file_name_list = output_file_name_list[:args.num_images]
    else:
        print(f"Number of images to generate is greater than the number of prompts. Generating only {len(prompts)} images and no sampling.")
        pass
    
    return prompts,output_file_name_list

def get_jacobi_param_dict(target_size, max_num_new_tokens, guidance_scale, 
                          seeds, image_top_k, text_top_k, prefix_token_sampler_scheme, 
                          local_chameleon_tokenizer_path, static_tree, multi_token_init_scheme):
    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 8)**2 -1,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg= True, #True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        local_chameleon_tokenizer_path = local_chameleon_tokenizer_path,
        static_tree = static_tree,
    )
    return jacobi_param_dict

def main(args):
    static_tree = args.static_tree
    tree_choices = args.tree_choices
    lantern_delta = args.lantern_delta
    groupsum_delta = args.groupsum_delta
    try:
        tree_choices = getattr(choices, args.tree_choices)
    except AttributeError:
        print(f"Tree choices {args.tree_choices} is not a valid choice")
        return
    # ******************** Args Initation ********************
    EMU_HUB = args.model_path
    target_size = args.target_size
    device = "cuda:0"
    local_chameleon_tokenizer_path=args.tokenizer_path
    output_path = args.output_path
    output_img_path = os.path.join(output_path,"img")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)

    # model path
    VQ_HUB = "BAAI/Emu3-VisionTokenizer"

    dtype = torch.bfloat16

    # ******************** Model Initation ********************
    # prepare model and processor
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation="sdpa", # "sdpa" , "flash_attention_2"
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=device, trust_remote_code=True).eval()
    image_tokenizer = image_tokenizer.to(dtype)

    # ******************** Input Initation ********************
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    seeds = [None, ]
    max_num_new_tokens = args.num_init_new_token
    multi_token_init_scheme = args.isp # 'repeat_horizon' random
    image_top_k = 2048
    text_top_k = 10
    guidance_scale = 3.0
    prefix_token_sampler_scheme = args.method
    # ******************** Load Benchmark ********************
    # image_area=model.config.image_area
    image_area = target_size **2
    assert image_area == target_size **2, f"Image area {image_area} does not match target size {target_size}"
    kwargs = dict(
        mode='G',
        ratio="1:1",
        image_area=image_area,
        return_tensors="pt",
    )
    # prepare hyper parameters
    GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=image_top_k,
        return_accl=True,
        # for static tree
        static_tree = static_tree,
        tree_choices = tree_choices,
        lantern_delta = lantern_delta,
        groupsum_delta = groupsum_delta,
    )
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    prompts,output_file_name_list = load_prompts(args)
    neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

    time_avg_forward = 0
    avg_acceptance_length = 0
    gen_count = 0
    with open(f"{output_path}/generation_configs.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    global_statistics = {}  

    # ******************** Generation Begin ********************
    for i, prompt in enumerate(prompts):
        # ******************** Input Begin ********************
        prompt += POSITIVE_PROMPT
        pos_inputs = processor(text=prompt, **kwargs)

        h, w = pos_inputs.image_size[0]

        pos_input_ids = pos_inputs.input_ids.to(device)
        neg_input_ids = neg_inputs.input_ids.to(device)

        jacobi_param_dict = get_jacobi_param_dict(target_size, max_num_new_tokens, guidance_scale, 
                          seeds, image_top_k, text_top_k, prefix_token_sampler_scheme, 
                          local_chameleon_tokenizer_path, static_tree,multi_token_init_scheme)
        jacobi_param_dict['h'] = h
        jacobi_param_dict['w'] = w
        jacobi_param_dict['neg_inputs'] = neg_input_ids
        jacobi_param_dict['classifier_free_guidance'] = guidance_scale

        from scheduler.jacobi_iteration_emu3 import renew_solver
        model, logits_processor = renew_solver(model, processor, **jacobi_param_dict)

        # generate
        model_inputs = model.prepare_batch_cfg_model_inputs(
            pos_input_ids, 
            neg_input_ids=neg_input_ids, 
            attention_mask=None,
        )
        pos_input_ids = model_inputs['pos_input_ids']
        attention_mask = model_inputs['attention_mask']
        # ******************** Generate Begin ********************
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                result = model.generate(
                    pos_input_ids,
                    GENERATION_CONFIG,
                    logits_processor=logits_processor, 
                    attention_mask=attention_mask,
                    neg_input_ids=neg_input_ids,
                )

        outputs = result.input_ids[0]
        time_forward = result.time_forward
        token_gen_len = result.token_gen_len
        loop_num = result.loop_num
        acceptance_length = token_gen_len / loop_num
        avg_acceptance_length += acceptance_length
        # ******************** Generate Saving ********************

        with torch.no_grad():
            mm_list = processor.decode(outputs)
        mm_list[1].save(os.path.join(output_img_path, f"{output_file_name_list[i]}.png"))
        statistics = {
            "prompt": prompt,
            "time": time_forward,
            "acceptance_length": acceptance_length,
            "loop_num": loop_num,
            "ann_id": output_file_name_list[i]
        }
        global_statistics[f"prompt_{i}"] = statistics
        time_avg_forward += time_forward
        with open(f"{args.output_path}/result_{args.slice}.json", "w") as f:
                json.dump(global_statistics, f, indent=4)
        gen_count += 1
    avg_acceptance_length = avg_acceptance_length/gen_count
    time_avg_forward = time_avg_forward/gen_count
    statistics = {
        "method":f"{prefix_token_sampler_scheme}_{multi_token_init_scheme}_{max_num_new_tokens}",
        "avg_acceptance_length":avg_acceptance_length,
        "time_forward_avg":time_avg_forward,
    }
    global_statistics[f"summary"] = statistics
    with open(f"{args.output_path}/result_{args.slice}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)
    print("Average time per generation: ", time_avg_forward)
    gc.collect()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="BAAI/Emu3-Gen",type=str, help="location of fake images for evaluation")
    parser.add_argument("--tokenizer_path", default='/data/lei/localmodel/lumina_mgpt/chameleon/tokenizer',type=str, help="location of the reference images for evaluation")
    parser.add_argument("--output_path", default='/home/leihaodong/ICLR25/exp/ablation',type=str)
    parser.add_argument("--target_size", type=int, default=720)

    parser.add_argument("--isp", default='random', type=str,help="repeat_horizon, random")
    parser.add_argument("--method", default='speculative_jacobi', type=str,help="'jacobi', 'speculative_jacobi'")
    parser.add_argument("--num_init_new_token", type=int, default=16)
    #Benchmark
    parser.add_argument("--benchmark_way", default='order', type=str, help="order or sample",)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="Atlantis, the most Fantasy high-quality photos")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=2)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    #Tree
    parser.add_argument("--static_tree", action="store_true", help="Enable static tree structure for draft token generation")
    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    
    #lantern
    parser.add_argument("--lantern_delta", type=int, help="Delta for LANTERN",
                        default=3)
    
    #groupsum
    parser.add_argument("--groupsum_delta", type=float, help="Delta for groupsum",
                        default=0.01)
    
    return parser

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)