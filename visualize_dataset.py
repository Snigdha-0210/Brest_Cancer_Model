import matplotlib.pyplot as plt
from dataset import BreastUltrasoundDataset
import numpy as np
import torch

def show_samples():
    dataset = BreastUltrasoundDataset(root_dir='.', split='train')
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    for i in range(3):
        # get random sample
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        # convert tensor back to numpy for visualization
        if torch.is_tensor(image):
            image = image.numpy().transpose(1, 2, 0)
        if torch.is_tensor(mask):
            mask = mask.numpy().squeeze()
            
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image {idx}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask {idx}')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    print("Saved samples to dataset_samples.png")

if __name__ == '__main__':
    show_samples()
