# **C**oncept **A**ttribution for **D**iffusion Models (CAD) Framework
We introduce a framework to compute the attribution of model parameters to a concept, allowing erase or amplify knowledge in diffusion models.
![alt text](CAD.png)
## Installation
Install from the requirements:
```
conda env create -f environment.yaml
```

## Erase nudity
Run this command to compute the gradient of the erasing objective, which is necessary to compute the attribution score:
```
python compute_positive_neurons.py --prompt naked --base_prompt '' --model_name $model_id
```
where ```model_id``` is the id of the diffusion model you are using, e.g. ```CompVis/stable-diffusion-v1-4```.

To evaluate, run:
```
python benchmarking/nudity_eval.py --eval_dataset i2p --ablate_ratio $ablate_ratio --model_id $model_id --target naked
```

We evaluate the quality on retained knowledge by comparing CLIP-Score and FID on COCO-30k dataset. Run the following command to generate images corresponding to COCO prompts and compute CLIP-Score

```
python benchmarking/clip_score.py  --model_id $model_id --target naked --batch_size 16 --ablate_ratio $ablate_ratio
```

After that you can compute FID with ```pytorch-fid```

```
python -m pytorch_fid path_to_coco path_to_generated_data
```

## Amplify nudity
To compute the gradient, run:
```
python compute_negative_neurons.py --prompt naked --base_prompt '' --model_name $model_id
```

To evaluate, run ```nudity_eval.py``` with the argument ```--cad_mode amplify```.

## Erase object
Run this command to compute the gradient of the erasing objective, which is necessary to compute the attribution score:
```
python compute_positive_neurons.py --prompt "${obj}" --base_prompt '' --model_name $model_id
```
where ```obj``` is the object you want to remove, e.g. ```parachute```, ```model_id``` is the id of the diffusion model you are using, e.g. ```CompVis/stable-diffusion-v1-4```.

To evaluate, run

```
python benchmarking/object_eval.py --target "${obj}" --test_mode erase --cad_mode erase --gpu $gpu --ablate_ratio $ablate_ratio --model_id $model_id
```

to compute the accuracy on the erased object and

```
python benchmarking/object_eval.py --target "${obj}" --test_mode keep --cad_mode erase --gpu $gpu --ablate_ratio $ablate_ratio --model_id $model_id
```

to compute the accuracy on other objects.

## Amplify object
To amplify knowledge about a object, you need a set of real images of that object in ```real_images/$object_name```. To compute the gradient, run 

```
python compute_negative_neurons.py --target "${obj}" --model_name $model_id --real_data
```
To evaluate, run the ```object_eval.py``` file with the argument ```--cad_mode amplify```.

## Erase art style

Run this command to compute the gradient of the erasing objective, which is necessary to compute the attribution score:
```
python compute_positive_neurons.py --prompt "${artist}" --base_prompt '' --model_name $model_id --concept_type art
```

Next, we generate images from the erased model

```
python benchmarking/generate_art.py --target "${artist}" --ablate_ratio $ratio --model_id $model_id --prompt_path prompts/big_artist_prompts.csv 
```

and compute LPIPS 
```
python benchmarking/lpips_eval.py --csv_path prompts/big_artist_prompts.csv --original_path $path_to_original_model_generated_data --edited_path $path_to_erased_model_generated_data
```
## Citation
Please cite our paper, as below, if you find the repository helpful:
```
@article{nguyen2024unveiling,
  title={Unveiling Concept Attribution in Diffusion Models},
  author={Nguyen, Quang H and Phan, Hoang and Doan, Khoa D},
  journal={arXiv preprint arXiv:2412.02542},
  year={2024}
}
```
