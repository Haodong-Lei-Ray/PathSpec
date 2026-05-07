import os
import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

import time
import argparse

from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
from llamagen.language.t5 import T5Embedder
from llamagen.llamagen import GPT_models
from llamagen.llamagen_solver import LlamaGenSolver, renew_llamagen, generate
from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler

from PIL import Image

import json
# Prompt
import random
import numpy as np

import json, csv
import re



os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_jacobi_param_dict(args):
    target_size = 512

    seeds = [None, ]
    max_num_new_tokens = args.num_init_new_token # 16
    multi_token_init_scheme = args.isp
    image_top_k = 1000
    text_top_k = 10
    guidance_scale = 7.5
    prefix_token_sampler_scheme = args.method

    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 16)**2 - max_num_new_tokens - 2, 
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        local_chameleon_tokenizer_path=args.tokenizer_path
    )
    return jacobi_param_dict


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
        coco = COCO("/data/lei/dataset/mscoco/annotations/captions_val2017.json")
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
    # ******************** Input Initation ********************
    max_num_new_tokens = args.num_init_new_token # 16
    prefix_token_sampler_scheme = args.method # 'jacobi', 'speculative_jacobi'
    multi_token_init_scheme = args.isp
    
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(gpt_model.__class__)

    jacobi_param_dict = get_jacobi_param_dict(args)
    image_top_k = jacobi_param_dict['image_top_k']

    gpt_model.__class__ = renew_llamagen(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)
    gpt_model.__class__ = renew_sampler(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    if not os.path.exists(args.t5_path):
        os.makedirs(args.t5_path)

    assert os.path.exists(args.t5_path), f"t5 model path {args.t5_path} does not exist"
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    
    # ******************** Load Benchmark ********************
    prompts, output_file_name_list = load_prompts(args)

    global_statistics = {}
    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    solver = LlamaGenSolver(
        model = gpt_model,
        image_top_k=image_top_k,
        image_top_p=args.top_p
    )
    print(f"start sampling...")
    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    qzshape = [1, args.codebook_embed_dim, latent_size, latent_size]
    
    time_avg = 0
    avg_acceptance_length = 0
    index_sample = None
    samples = None
    gen_count = 0
    for i in range(len(c_indices)):
        t1 = time.time()
        result = solver.generate(
            c_indices[i:i+1], latent_size ** 2, 
            c_emb_masks[i:i+1], 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=image_top_k,
            top_p=args.top_p, sample_logits=True,
            return_accl=True
        )
        sampling_time = time.time() - t1
        if index_sample == None:
            index_sample = result.input_ids
        else:
            index_sample = torch.concat([index_sample,result.input_ids], dim=0)
        token_gen_len = result.token_gen_len
        loop_num = result.loop_num
        acceptance_length = token_gen_len / loop_num
        avg_acceptance_length += acceptance_length
        time_avg += sampling_time
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
        statistics = {
            "prompt": prompts[i],
            "time": sampling_time,
            "acceptance_length": acceptance_length,
            "loop_num": loop_num,
            "ann_id": output_file_name_list[i]
            }
        global_statistics[f"prompt_{i}"] = statistics
        gen_count += 1
        
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    images = samples
    images = images.clamp(min=-1, max=1)
    images_sum = (images - images.min()) / (images.max() - images.min()) * 255

    # TODO: 修改你的本地输出地址
    output_path = args.output_path
    output_img_path = os.path.join(output_path,"img")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)

    for i in range(len(c_indices)):
        images = images_sum[i].permute(1, 2, 0).cpu().numpy()
        result_image = Image.fromarray((images).astype("uint8"))
        output_file_name = str(output_file_name_list[i]) + ".png"
        result_image.save(os.path.join(output_img_path,output_file_name))
        print(f"image is saved to sample_{os.path.join(output_img_path,output_file_name)}.png")
        avg_acceptance_length = avg_acceptance_length/gen_count
        time_avg_forward = time_avg/gen_count
    statistics = {
        "method":f"{prefix_token_sampler_scheme}_{multi_token_init_scheme}_{max_num_new_tokens}",
        "avg_acceptance_length":avg_acceptance_length,
        "time_forward_avg":time_avg_forward,
    }
    global_statistics[f"summary"] = statistics
    with open(f"{args.output_path}/result_{args.slice}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)
    print("Average time per generation: ", time_avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5_path", type=str, default='/data/lei/localmodel/')
    parser.add_argument("--t5_model_type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5_feature_max_len", type=int, default=120)
    parser.add_argument("--t5_feature_dim", type=int, default=2048)
    parser.add_argument("--no_left_padding", action='store_true', default=False)
    parser.add_argument("--gpt_model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt_ckpt", type=str, default="/home/leihaodong/pretrained_models/t2i_XL_stage2_512.pt")
    parser.add_argument("--gpt_type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls_token_num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq_model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq_ckpt", type=str, default="/home/leihaodong/pretrained_models/vq_ds16_c2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample_size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value to sample with")
    
    parser.add_argument("--tokenizer_path", default='/data/lei/localmodel/lumina_mgpt/chameleon/tokenizer',type=str, help="location of the reference images for evaluation")
    
    # SJD
    parser.add_argument("--isp", default='random', type=str,help="repeat_horizon, random")
    parser.add_argument("--method", default='speculative_jacobi', type=str,help="'jacobi', 'speculative_jacobi'")
    parser.add_argument("--num_init_new_token", type=int, default=16)
    
    # Benchmark
    parser.add_argument("--output_path", default='/home/leihaodong/AAAI25/exp/FSJD',type=str)
    parser.add_argument("--benchmark_way", default='order', type=str, help="order or sample",)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="Atlantis, the most Fantasy high-quality photos")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=2)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    
    args = parser.parse_args()
    main(args)