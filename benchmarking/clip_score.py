import os
import sys
import json
import torch
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion import safety_checker
import clip
from torchvision.transforms import Compose, Resize, Normalize
from utils import ablate_model, apply_model
import torchvision

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion')
    parser.add_argument('--removal_mode', type=str, default=None, choices=['erase', 'keep'])
    parser.add_argument('--benchmarking_result_path', type=str, default='results')
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--ablate_ratio', type=float, default=0)
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--objective', type=str, default='loss_detach')
    parser.add_argument('--ablate_type', type=str, default='positive')
    parser.add_argument('--flatten_pruning', action='store_true')
    parser.add_argument('--training_loss', action='store_true')
    parser.add_argument('--cad_mode', type=str, default='erase', choices=['erase', 'amplify'])
    return parser.parse_args()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.dataset = data['prompt']
        self.seeds = data['evaluation_seed']
        self.img_id = data['image_id']
        print(f"Number of prompts: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]
        seed = self.seeds[idx]
        id = self.img_id[idx]
        return prompt, seed, id

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, args.target, 'coco', f'concept_{args.ablate_ratio}')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    if not os.path.exists(f'{args.benchmarking_result_path}'):
        os.makedirs(f'{args.benchmarking_result_path}')

    prompts = pd.read_csv(f'prompts/coco_30k.csv')
    dataloader = torch.utils.data.DataLoader(CustomDataset(prompts), batch_size=args.batch_size, shuffle=False)
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to(args.gpu)
    model.set_progress_bar_config(disable=True)
    def sc(self, clip_input, images):
        return images, [False for i in images]

    safety_checker.StableDiffusionSafetyChecker.forward = sc
    safety_checker_ = safety_checker.StableDiffusionSafetyChecker

    model = apply_model(model, args.model_type, args.target, f'cuda:{args.gpu}')

    if args.ablate_ratio > 0:
        grad_dict = torch.load(f'grad/{args.model_id}/{args.model_type}/{args.cad_mode}/{args.target}_grad.pth', map_location=f'cuda:{args.gpu}')
        model = ablate_model(model, grad_dict, args.ablate_ratio, ablate_type=args.ablate_type, flatten_pruning=args.flatten_pruning)


    clip_model, transform = clip.load("ViT-B/32", device=f'cuda:{args.gpu}', jit=False)
    clip_model.eval() 
    clip_transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            # lambda image: image.convert("RGB"),
            # ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    

    avg_acc = 0
    n_samples = 0
    score = 0
    with torch.no_grad():
        for iter, (prompt, seed, ids) in enumerate(pbar := tqdm(dataloader)):
            if args.dbg and iter > 10:
                break

            prompt = list(prompt)
            generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in seed.numpy()]
            images = model(prompt, generator=generator, safety_checker=safety_checker_, output_type="pt").images
            for i, image in enumerate(images):
                torchvision.utils.save_image(image, os.path.join(args.benchmarking_result_path, f'{ids[i]}.png'))
            images = clip_transform(images)
            img_feats = clip_model.encode_image(images)
            c_data = clip.tokenize(prompt, truncate=True).to(f'cuda:{args.gpu}')
            text_feats = clip_model.encode_text(c_data)
            score += torch.nn.CosineSimilarity()(text_feats, img_feats).sum().item()
            n_samples += len(prompt)
            pbar.set_description(f'Clip score: {score / n_samples}')

    results = {'Clip score': score / n_samples}
    with open(os.path.join(args.benchmarking_result_path, f"clip_score.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()