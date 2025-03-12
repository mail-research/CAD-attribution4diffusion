import os
import sys
import json
import torch
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from diffusers import StableDiffusionPipeline
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion import safety_checker
from utils import ablate_model, apply_model
import torchvision


def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion')
    parser.add_argument('--test_mode', type=str, default=None, choices=['erase', 'keep'])
    parser.add_argument('--benchmarking_result_path', type=str, default='results')
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--ablate_ratio', type=float, default=0.001)
    parser.add_argument('--objective', type=str, default='loss_detach')
    parser.add_argument('--ablate_type', type=str, default='positive')
    parser.add_argument('--flatten_pruning', action='store_true')
    parser.add_argument('--cad_mode', type=str, default='erase', choices=['erase', 'amplify'])
    parser.add_argument('--data_type', type=str, default='generated')
    return parser.parse_args()

# Dataset class to test concept erasure
class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.prompts = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']

        # select only prompts that have the concept to remove
        self.prompts = [(self.prompts[i], self.seeds[i], concepts_to_remove) for i in range(len(self.prompts)) if concepts_to_remove.lower() == self.labels[i].lower()]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    
# Dataset class to test concept keeping
class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.dataset = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']
        self.dataset = [(self.dataset[i], self.seeds[i], self.labels[i].lower()) for i in range(len(self.dataset)) if concepts_to_remove.lower() != self.labels[i].lower()]
        print(f"Number of prompts: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx][0]
        seed = self.dataset[idx][1]
        label = self.dataset[idx][2].lower()
        return prompt, seed, label
    

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, args.target, 'benchmarking', f'concept_{args.test_mode}_{args.ablate_ratio}_{args.ablate_type}')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    if not os.path.exists(f'{args.benchmarking_result_path}'):
        os.makedirs(f'{args.benchmarking_result_path}')

    # Load dataset
    data = pd.read_csv(f'prompts/imagenette.csv')
    if args.test_mode == 'erase':
        dataloader = torch.utils.data.DataLoader(CustomDatasetErasure(data, args.target), batch_size=args.batch_size, shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(CustomDatasetKeep(data, args.target), batch_size=args.batch_size, shuffle=False)

    print("Number of prompts: ", len(dataloader.dataset))

    # Load the concept erased model
    remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(args.gpu)
    remover_model.set_progress_bar_config(disable=True)
    def sc(self, clip_input, images):
        return images, [False for i in images]

    safety_checker.StableDiffusionSafetyChecker.forward = sc
    safety_checker_ = safety_checker.StableDiffusionSafetyChecker

    remover_model = apply_model(remover_model, args.model_type, args.target, f'cuda:{args.gpu}')

    if args.ablate_ratio > 0:
        grad_dict = torch.load(f'grad/{args.model_id}/{args.model_type}/{args.cad_mode}/{args.target}_grad.pth', map_location=f'cuda:{args.gpu}')
        remover_model = ablate_model(remover_model, grad_dict, args.ablate_ratio, ablate_type=args.ablate_type, flatten_pruning=args.flatten_pruning)

    # Pre-trained ResNet50 
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to(args.gpu)
    classifier.eval()

    preprocess = weights.transforms()

    # test model on dataloader
    avg_acc = 0
    total = 0
    for iter, (prompt, seed, label) in enumerate(pbar := tqdm(dataloader)):
        if args.dbg and iter > 10:
            break

        prompt = list(prompt)
        generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in seed.numpy()]

        removal_images = remover_model(prompt, generator=generator, safety_checker=safety_checker_ , output_type="pt").images

        torchvision.utils.save_image(removal_images, os.path.join(args.benchmarking_result_path, f'{iter}.png'))
        image = preprocess(removal_images)
        image = image.to(args.gpu)
        with torch.no_grad():
            output = classifier(image)
        for i, o in enumerate(output):
            s, indices = torch.topk(o, 1)
            indices = indices.cpu().numpy()
            pred_labels = [weights.meta["categories"][idx] for idx in indices]
            pred_labels = [l.lower() for l in pred_labels]
            if label[i] in pred_labels:
                avg_acc += 1
        total += len(prompt)
        pbar.set_description(f'Acc: {avg_acc / total}')

    print("Object predicted in: %d/%d images" % (avg_acc, len(dataloader.dataset)))
    print(f"Average accuracy: {avg_acc / len(dataloader.dataset)}")
    results = {"average_accuracy": avg_acc / len(dataloader.dataset)}
    with open(os.path.join(args.benchmarking_result_path, f"results_{args.test_mode}.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()

        



