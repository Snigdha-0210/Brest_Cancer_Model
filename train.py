import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BreastUltrasoundDataset
from models import UNet, UNetGCN
from utils import BCEDiceLoss, calculate_metrics

def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = BreastUltrasoundDataset(root_dir=args.data_dir, split='train', transform=train_transform)
    valid_dataset = BreastUltrasoundDataset(root_dir=args.data_dir, split='valid', transform=val_transform)
    
    # Num workers set to 0 for better compatibility on Windows unless specified
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        model = UNetGCN(in_channels=3, out_channels=1, num_nodes=args.num_nodes).to(device)
        
    criterion = BCEDiceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
        
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # Calculate metrics
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v * images.size(0)
                    
        val_loss /= len(valid_loader.dataset)
        for k in val_metrics:
            val_metrics[k] /= len(valid_loader.dataset)
            
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
              
        scheduler.step(val_metrics['dice'])
        
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            if args.model == 'unet_gcn':
                save_name = f"best_model_{args.model}_nodes_{args.num_nodes}.pth"
            else:
                save_name = f"best_model_{args.model}.pth"
            torch.save(model.state_dict(), os.path.join(args.save_dir, save_name))
            print("Saved new best model!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='Dataset root directory')
    parser.add_argument('--model', type=str, choices=['unet', 'unet_gcn'], default='unet', help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_nodes', type=int, default=64, help='Number of nodes for NSCGCN')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save weights')
    
    args = parser.parse_args()
    main(args)
