import os
import sys
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from utils import ablate_model, apply_model


# Disable safety checker completely
def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default='naked')
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--benchmarking_result_path', type=str, default='results')
    parser.add_argument('--ablate_ratio', type=float, default=0.001)
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--ablate_type', type=str, default='positive')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--training_loss', action='store_true')
    parser.add_argument('--objective', type=str, default='loss_detach')
    parser.add_argument('--flatten_pruning', action='store_true')
    parser.add_argument('--cad_mode', type=str, default='erase', choices=['erase', 'amplify'])
    return parser.parse_args()


def main():
    args = input_args()
    print("Arguments: ", args.__dict__)
    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, 'art', args.target, f'{args.ablate_ratio}', 'generated_images')
    print("Benchmarking result path: ", args.benchmarking_result_path)


    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)
    
    # make a dataloader of prompts
    prompts = pd.read_csv(args.prompt_path)

    # Load the concept erased model
    remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(args.gpu)
    remover_model.set_progress_bar_config(disable=True)

    remover_model = apply_model(remover_model, args.model_type, args.target, f'cuda:{args.gpu}')
    if args.ablate_ratio > 0:
        grad_dict = torch.load(f'grad/{args.model_id}/{args.model_type}/{args.cad_mode}/{args.target}_grad.pth', map_location=f'cuda:{args.gpu}')
        remover_model = ablate_model(remover_model, grad_dict, args.ablate_ratio, ablate_type=args.ablate_type, flatten_pruning=args.flatten_pruning)


    for i, item in (pbar := tqdm(prompts.iterrows())):
        prompt = item['prompt']
        seed = item['evaluation_seed']
        if os.path.exists(os.path.join(args.benchmarking_result_path, f"{i}_original.png")):
            print(f"Skipping iteration {i}")
            continue
        else:
            # get the image after removing the concept
            torch.manual_seed(seed)
            np.random.seed(seed)
            removal_image = remover_model(prompt, safety_checker=safety_checker_).images[0]

            # save images
            os.makedirs(os.path.join(args.benchmarking_result_path, item['artist']), exist_ok=True)
            removal_image.save(os.path.join(args.benchmarking_result_path, item['artist'], f"{i}_removed.png"))

if __name__ == '__main__':
    main()