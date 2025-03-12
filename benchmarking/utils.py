import torch

def apply_model(model, model_type, target, device):
    if model_type is None:
        return model
    elif model_type == 'esd':
        name = ''.join(target.split())
        esd_path = f'esd_checkpoint/compvis-word_{name}-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_{name}-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05.pt'
        model.unet.load_state_dict(torch.load(esd_path))
    elif model_type == 'uce':
        path = f'uce_checkpoint/erased-{target.lower()}-towards_art-preserve_true-sd_1_4-method_replace.pt'
        model.unet.load_state_dict(torch.load(path, map_location='cpu'))
    elif model_type == 'rece':
        name = ''.join(target.split()).lower()
        path = f'rece_checkpoint/{name}.pt'
        model.unet.load_state_dict(torch.load(path, map_location='cpu'))
    elif model_type == 'conceptprune':
        model.unet.load_state_dict(torch.load(f'concept_prune_{target}_checkpoint/skill_ratio_0.01_timesteps_9_threshold_0.5.pt', map_location='cpu'))
    else:
        model.unet.load_state_dict(torch.load(f'ckpts/{model_type}/{target}.pt', map_location=device))
    
    return model

def ablate_model(model, grad_dict, ratio, ablate_type='positive', flatten_pruning=False):
    for name, mod in model.unet.named_parameters():
        if len(mod.shape) == 2 and 'ff' in name:
            weight = mod.data * grad_dict[name]
            
            if flatten_pruning:
                shape = weight.shape
                weight = weight.reshape(1, -1)
            mask = torch.zeros_like(weight)
            sorted_idx = torch.argsort(weight, dim=-1, descending=True)
            if ablate_type == 'positive':
                selected_idx = sorted_idx[:, int(ratio * weight.shape[1]):]
            elif ablate_type == 'both':
                start = int(ratio * weight.shape[1])
                end = weight.shape[1] - int(ratio *  weight.shape[1])
                selected_idx = sorted_idx[:, start:end]
            else:
                end = weight.shape[1] - int(ratio *  weight.shape[1])
                selected_idx = sorted_idx[:, :end]
            mask.scatter_(1, selected_idx, 1)
            if ablate_type == 'positive':
                mask = mask.bool() + (weight < 0)
            elif ablate_type == 'negative':
                mask = mask.bool() + (weight > 0)
            if flatten_pruning:
                mask = mask.reshape(shape)
            mod.data = mod.data * mask
    return model