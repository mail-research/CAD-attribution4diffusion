import torch
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image

import argparse
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import safety_checker
from torchvision import transforms


def get_score(model, latents, prompt, device, ddim_steps=50, n_est=1):

    text_input = model.tokenizer(
        [prompt], padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0]
    grad_dict = {}
    for name, mod in model.unet.named_parameters():                                                                                                                             
        grad_dict[name] = 0 

    objs = 0
    for i, t in enumerate(pbar := tqdm(range(1, ddim_steps+1))):
        for _ in range(n_est):
            for index in range(len(latents)):
                latent = latents[index].unsqueeze(0)
                noise = torch.randn_like(latent)
                timestep = model.scheduler.timesteps[-t].long()
                z = model.scheduler.add_noise(latent, noise, timestep)
                noise_pred = model.unet(z, timestep, encoder_hidden_states=text_embeddings).sample
                obj = ((noise_pred - noise) ** 2).mean() 
                objs += obj.item()
                model.unet.zero_grad()
                obj.backward()
                pbar.set_description(f'Obj: {obj.item()}, mean: {objs / (_ + 1)}')
                for name, mod in model.unet.named_parameters():                                                                                                                             
                    grad_dict[name] += mod.grad / n_est

    return grad_dict

def main(args):
    model = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16).to(args.device)
    model.set_progress_bar_config(disable=True)
    model.requires_safety_checker = False
    def sc(self, clip_input, images):
        return images, [False for i in images]

    safety_checker.StableDiffusionSafetyChecker.forward = sc
    safety_checker_ = safety_checker.StableDiffusionSafetyChecker

    if args.real_data:
        data_path = f"real_images/{args.target}"
        transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        for filename in os.listdir(data_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                images.append(transform(Image.open(os.path.join(data_path, filename))))
        images = torch.stack(images)

        latents = []
        bsz = 5
        start = 0
        with torch.no_grad():
            for _ in tqdm(range((images.shape[0] - 1) // bsz + 1)):
                img = images[start:start+bsz].to(args.device).to(torch.float16)
                latent = model.vae.encode(img)[0].sample() * model.vae.config.scaling_factor
                latents.append(latent)
                start += img.shape[0]
        latents = torch.cat(latents)
    else:
        latents = []
        bsz = 5
        prompt_dict = {'nudity': 'naked', 'object': f'a photo of {args.target}', 'style': f'a picture in the style of {args.target}'}
        with torch.no_grad():
            for _ in tqdm(range(args.ddim_steps // bsz)):
                img = model([prompt_dict[args.concept_type]], output_type='pt', num_images_per_prompt=bsz, safety_checker=safety_checker_).images
                latent = model.vae.encode(img)[0].sample()
                latents.append(latent)
        latents = torch.cat(latents)

    if args.model_type is not None:
        model.unet.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    
    model.scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    model.scheduler.set_timesteps(50)

    os.makedirs(f'grad/{args.model_name}/{args.model_type}/', exist_ok=True)

    grad_dict = get_score(model, latents, args.target, args.device, args.ddim_steps, n_est=args.n_est)
    torch.save(grad_dict, f'grad/{args.model_name}/{args.model_type}/amplify/{args.target}_grad.pth') 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CAD-Amplify',
                    description = 'Compute the attribution score for CAD-Amplify')
    parser.add_argument('--target', help='prompt corresponding to target concept', type=str, required=True)
    parser.add_argument('--device', help='cuda devices', type=str, required=False, default='0')
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--model_name', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--model_type', help='whether the model is the base SD or erased by other algorithms', type=str, default=None)
    parser.add_argument('--ckpt_path', help='the path to the checkpoint of the pretrained model, if model_type is not None', type=str, default=None)
    parser.add_argument('--real_data', action='store_true')
    parser.add_argument('--n_est', type=int, default=1)
    args = parser.parse_args()

    main(args)
