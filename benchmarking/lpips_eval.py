from __future__ import print_function
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import argparse
import lpips
import warnings
warnings.filterwarnings("ignore")

# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def is_target_artist(artist_name, erased_artist_name):
    return erased_artist_name.lower() in artist_name.lower() 

def image_loader(image_name):
    try:
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)  # Assuming 'loader' is defined somewhere as a transform
        image = (image-0.5)*2
        return image.to(torch.float)
    except Exception as e:
        print(f"Failed to load image {image_name}: {e}")
        raise


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(prog='LPIPS', description='Calculate LPIPS between two images')
    parser.add_argument('--original_path', help='Path to original image', type=str, required=False)
    parser.add_argument('--edited_path', help='Path to edited image', type=str, required=False)
    parser.add_argument('--csv_path', help='Path to CSV file containing prompts', type=str, required=True)
    parser.add_argument('--save_path', help='Path to save results', type=str, default=None)
    parser.add_argument('--model_type', help='Type of model to use', type=str, default='alex')

    import lpips
    print(lpips)
    loss_fn_alex = lpips.LPIPS(net='alex')
    args = parser.parse_args()

    artists = ["Pablo Picasso", "Van Gogh", "Rembrandt", "Andy Warhol", "Caravaggio"]
    artists_2 = ["Pablo Picasso", "Van Gogh", "Rembrandt", "Andy Warhol", "Caravaggio"]
    
    for i in range(len(artists_2)):
        scores = []
        other_artist_scores = []  # To track scores for other artists
        erased_artist = artists_2[i] 
        print(f"Processing Erased Artist: {erased_artist}")
        
        for j in range(len(artists)):
            artist = artists[j]
            score, total = 0, 0
            
            file_names = [name for name in os.listdir(args.original_path) if name.endswith('.png')]

            for file_name in file_names:
                try:
                    original = image_loader(os.path.join(args.original_path, file_name))
                    edited = image_loader(os.path.join(args.edited_path, file_name))

                    l = loss_fn_alex(original, edited)
                    score += l.item()
                    total += 1

                except Exception as e:
                    print(f'Failed to process file {file_name}: {e}')
            
            if total > 0:
                average_score = score / total
                
                if is_target_artist(artist, erased_artist):
                    target_artist_score = average_score
                    print(f'    Target Artist: {artist}, LPIPS Score: {average_score}')
                else:
                    other_artist_scores.append(average_score)  
                    print(f'    Other Artist: {artist}, LPIPS Score: {average_score}')
            
        if other_artist_scores:
            print(len(other_artist_scores))
            other_artists_average = np.mean(other_artist_scores)
            print(f"Average LPIPS for the four other artists: {other_artists_average}")
        
