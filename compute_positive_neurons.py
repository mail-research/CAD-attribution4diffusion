import torch
import os
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionPipeline
from diffuser_custom_generate import custom_sample
from diffusers import DDIMScheduler

def get_score(model, prompt, base_prompt, start_guidance, device, ddim_steps=50, n_est=1):
    text_input = model.tokenizer(
        [prompt, base_prompt], padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0]
    grad_dict = {}
    for name, mod in model.unet.named_parameters():                                                                                                                             
        grad_dict[name] = 0 

    for t in tqdm(range(1, ddim_steps+1)):
        for _ in range(n_est):
            z, timestep = custom_sample(model, [prompt], until_t=t, guidance_scale=start_guidance)
            z = model.scheduler.scale_model_input(z, t)
            z = torch.cat([z] * 2)
            noise = model.unet(z, timestep, encoder_hidden_states=text_embeddings).sample
            
            obj = ((noise[0] - noise[1].detach()) ** 2).sum()

            model.unet.zero_grad()
            obj.backward()
            for name, mod in model.unet.named_parameters():                                                                                                                             
                grad_dict[name] += mod.grad

    return grad_dict


def main(args, prompt):
    model = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16).to(args.device)
    if args.model_type is not None:
        model.unet.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    
    model.scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    model.scheduler.set_timesteps(50)
    os.makedirs(f'grad/{args.model_name}/{args.model_type}/erase', exist_ok=True)

    if args.concept_type == 'art':
        prompt = f'a photo in the style of {prompt}'
    grad_dict = get_score(model, prompt, args.base_prompt, args.start_guidance, args.device, args.ddim_steps)
    torch.save(grad_dict, f'grad/{args.model_name}/{args.model_type}/erase/{args.prompt}_grad.pth') 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CAD-Erase',
                    description = 'Compute the attribution score for CAD-Erase')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--base_prompt', help='prompt corresponding to concept to erase', type=str, default='')
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--model_name', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--ckpt_path', help='the path to the checkpoint of the pretrained model, if model_type is not None', type=str, default=None)
    parser.add_argument('--concept_type', type=str, default='object')
    args = parser.parse_args()
    
    args.device = f'cuda:{args.device}'
    main(args, prompt=args.prompt)
