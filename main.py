import argparse
import os
import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
# print(sys.path)

import gc

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from PIL import Image
import torch
import time

import random
import numpy as np

import json, csv
import re

import lumina_mgpt.data.drafters.choices as choices

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
    elif args.prompt == "MSCOCO2017Val":
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

def main(args):
    static_tree = args.static_tree
    static_tree_plus = args.static_tree_plus
    tree_choices = args.tree_choices
    lantern_delta = args.lantern_delta
    groupsum_delta = args.groupsum_delta
    threshold = args.sjd_pp_threshold
    try:
        tree_choices = getattr(choices, args.tree_choices)
    except AttributeError:
        print(f"Tree choices {args.tree_choices} is not a valid choice")
        return
    # ******************** Args Initation ********************
    model_path = args.model_path
    target_size = args.target_size
    target_size_h, target_size_w = target_size, target_size
    device = "cuda:0"
    # TODO: 修改你的本地chameleon
    local_chameleon_tokenizer_path=args.tokenizer_path
    # TODO: 修改你的本地输出地址
    output_path = args.output_path
    output_img_path = os.path.join(output_path,"img")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)

    # ******************** Input Initation ********************
    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        precision="bf16",
        target_size=target_size,
        device = device,
        local_chameleon_tokenizer_path = local_chameleon_tokenizer_path
    )

    seeds = [None, ] #[_ for _ in range(124, 200) ]
    max_num_new_tokens = args.num_init_new_token # 16
    multi_token_init_scheme = args.isp # 'repeat_horizon' random
    image_top_k = 2000 
    text_top_k = 10
    guidance_scale = 3.0
    prefix_token_sampler_scheme = args.method # 'jacobi', 'speculative_jacobi'

    # ******************** Load Benchmark ********************

    prompts,output_file_name_list = load_prompts(args)

    template_condition_sentences = [
        f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n",
    ] * len(prompts)

    # ******************** Image Generation ********************
    from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
    # print(inference_solver.__class__)
    inference_solver = renew_pipeline_sampler(
        inference_solver,
        jacobi_loop_interval_l = 3,
        jacobi_loop_interval_r = (target_size // 16)**2 + target_size // 16 - 10,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        local_chameleon_tokenizer_path = local_chameleon_tokenizer_path,
        static_tree = static_tree,
        static_tree_plus = static_tree_plus
    )
    time_avg = 0
    time_avg_forward = 0
    avg_acceptance_length = 0
    gen_count = 0
    with open(f"{output_path}/generation_configs.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    global_statistics = {}  
    
    for seed in seeds:
        inference_solver.model.seed = seed
        for i, q_image_content_condition in enumerate(prompts):
            q1 = template_condition_sentences[i] + q_image_content_condition

            # output_file_name = model_path.split("/")[-1] + "-" + q_image_content_condition[:30] + '-' + str(max_num_new_tokens) + '-init-' + multi_token_init_scheme[:6] + '-seed' + str(seed) + '-img_topk' + str(image_top_k) + ".png"
            output_file_name = str(output_file_name_list[i]) + ".png"

            time_start = time.time()
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t1.record()

            result = inference_solver.generate(
                images=[],
                qas=[[q1, None]],
                max_gen_len=8192,
                temperature=1.0,
                logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k, static_tree = static_tree),
                return_accl=True,
                # for static tree
                static_tree = static_tree,
                tree_choices = tree_choices,
                lantern_delta = lantern_delta,
                groupsum_delta = groupsum_delta,
                threshold = threshold
            )
            generated = result.input_ids
            
            t2.record()
            torch.cuda.synchronize()

            t = t1.elapsed_time(t2) / 1000
            time_end = time.time()

            a1, new_image = generated[0], generated[1][0]

            result_image = inference_solver.create_image_grid([new_image], 1, 1)
            result_image.save(os.path.join(output_img_path,output_file_name))
            
            time_forward = result.time_forward
            token_gen_len = result.token_gen_len
            loop_num = result.loop_num
            acceptance_length = token_gen_len / loop_num
            avg_acceptance_length += acceptance_length
            statistics = {
                "prompt": q_image_content_condition,
                "time": time_forward,
                "acceptance_length": acceptance_length,
                "loop_num": loop_num,
                "Time elapsed cuda": t,
                "Time elapsed": time_end - time_start,
                "ann_id": output_file_name_list[i]
            }
            
            global_statistics[f"prompt_{i}"] = statistics
            time_avg += t / len(seeds)
            time_avg_forward += time_forward
            with open(f"{args.output_path}/result_{args.slice}.json", "w") as f:
                json.dump(global_statistics, f, indent=4)
            gen_count += 1
    avg_acceptance_length = avg_acceptance_length/gen_count
    time_avg = time_avg/gen_count
    time_avg_forward = time_avg_forward/gen_count
    statistics = {
        "method":f"{prefix_token_sampler_scheme}_{multi_token_init_scheme}_{max_num_new_tokens}",
        "avg_acceptance_length":avg_acceptance_length,
        "time_forward_avg":time_avg_forward,
        "time_avg":time_avg,
    }
    global_statistics[f"summary"] = statistics
    with open(f"{args.output_path}/result_{args.slice}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)
    print("Average time per generation: ", time_avg)
    del inference_solver
    gc.collect()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Alpha-VLLM/Lumina-mGPT-7B-768",type=str, help="location of fake images for evaluation")
    parser.add_argument("--tokenizer_path", default='/data/lei/localmodel/lumina_mgpt/chameleon/tokenizer',type=str, help="location of the reference images for evaluation")
    parser.add_argument("--output_path", default='/home/leihaodong/AAAI25/exp/FSJD',type=str)
    parser.add_argument("--target_size", type=int, default=768)

    parser.add_argument("--isp", default='random', type=str,help="repeat_horizon, random")
    parser.add_argument("--method", default='speculative_jacobi', type=str,help="'jacobi', 'speculative_jacobi'")
    parser.add_argument("--num_init_new_token", type=int, default=16)
    
    parser.add_argument("--benchmark_way", default='order', type=str, help="order or sample",)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="Atlantis, the most Fantasy high-quality photos")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=2)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    
    #Tree
    parser.add_argument("--static_tree", action="store_true", help="Enable static tree structure for draft token generation")
    parser.add_argument("--static_tree_plus", action="store_true", help="Enable static tree structure for draft token generation")
    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    
    #lantern
    parser.add_argument("--lantern_delta", type=int, help="Delta for LANTERN",
                        default=3)
    
    #groupsum
    parser.add_argument("--groupsum_delta", type=float, help="Delta for groupsum",
                        default=0.01)
    #sjd++
    parser.add_argument("--sjd_pp_threshold", type=float, help="Threshold for sjd++",
                        default=0.5)
    
    return parser

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)