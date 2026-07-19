import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import BreastUltrasoundDataset
from models import UNet, UNetGCN
from utils import calculate_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    val_transform = A.Compose([A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    test_dataset = BreastUltrasoundDataset(root_dir=args.data_dir, split='test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        model = UNetGCN(in_channels=3, out_channels=1, num_nodes=args.num_nodes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    test_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    if args.save_preds:
        model_name_dir = args.model if args.model == 'unet' else f'{args.model}_nodes_{args.num_nodes}'
        os.makedirs(os.path.join(args.data_dir, 'predictions', model_name_dir), exist_ok=True)
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
            images, masks = (images.to(device), masks.to(device))
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(preds, masks)
            for k, v in batch_metrics.items():
                test_metrics[k] += v * images.size(0)
            if args.save_preds:
                pred_mask = (preds[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                filename = test_dataset.images[i]
                model_name_dir = args.model if args.model == 'unet' else f'{args.model}_nodes_{args.num_nodes}'
                save_path = os.path.join(args.data_dir, 'predictions', model_name_dir, filename)
                cv2.imwrite(save_path, pred_mask)
    for k in test_metrics:
        test_metrics[k] /= len(test_loader.dataset)
    print(f'Final Test Metrics for {args.model}:')
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='Dataset root directory')
    parser.add_argument('--model', type=str, choices=['unet', 'unet_gcn'], required=True, help='Model to test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_nodes', type=int, default=64, help='Number of nodes for NSCGCN')
    parser.add_argument('--save_preds', action='store_true', help='Save prediction masks')
    args = parser.parse_args()
    main(args)
