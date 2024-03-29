import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import os

running_task = 'bet_alt' # vent_alt csf_alt gm_alt wm_alt bet_alt ; vent_sag_alt

SOURCE_DIR = '../data/processed/'

dir_img = Path(SOURCE_DIR + 'sorted_imgs_no_brain_extraction/train/') # ct_norm_sag
dir_mask = Path(SOURCE_DIR + 'sorted_masks/train')

dir_checkpoint = Path('./checkpoints_' + running_task)

dir_img_val = Path(SOURCE_DIR + 'sorted_imgs_no_brain_extraction/val/') # ct_norm_sag
dir_mask_val = Path(SOURCE_DIR + 'sorted_masks/val/')

def write_summary(epoch, lr, step, train_loss, val_dice, file_path='summary.txt'):
    with open(file_path, 'a') as file:
        file.write(f"{epoch}, {lr}, {step}, {train_loss}, {val_dice}\n")

# Function to log training information
def log_training_info(info, log_path='training_logs.txt'):
    with open(log_path, 'a') as file:
        file.write(f"{info}\n")

import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)


def train_net(net, device, epochs=5, batch_size=1, learning_rate=1e-5, img_scale=1.0, amp=False, save_checkpoint=True):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale)
    n_train = len(dataset)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    if wandb.run is None:  # Ensure wandb session is not duplicated
        wandb.init(project='U-Net', config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_scale": img_scale,
            "amp": amp
        })

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001, path=dir_checkpoint / 'early_stopping_model.pth')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            images = batch['image'].to(device, dtype=torch.float32)
            true_masks = batch['mask'].to(device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)
                if hasattr(net, 'n_classes'):  # If your model supports dice_loss calculation
                    loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                                      F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                      multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()

        # Calculate validation dice score
        val_dice = evaluate(net, val_loader, device)

        logging.info(f'Epoch {epoch+1}, LR: {optimizer.param_groups[0]["lr"]}, Step: {len(train_loader)*(epoch+1)}, '
                     f'Train Loss: {epoch_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}')

        # Update learning rate
        scheduler.step(val_dice)

        # Early Stopping check
        early_stopping(val_dice, net)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Save model checkpoint if not early stopping
        if save_checkpoint and not early_stopping.early_stop:
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch{epoch+1}.pth'
            torch.save(net.state_dict(), checkpoint_path)
            logging.info(f'Checkpoint saved at {checkpoint_path}')

        # Log metrics to wandb
        wandb.log({"epoch": epoch+1, "learning_rate": optimizer.param_groups[0]["lr"],
                   "train_loss": epoch_loss/len(train_loader), "val_dice": val_dice})


def get_args():
    
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    
    parser.add_argument("--gpu", type=str, default='0', help='choose which gpu(s) to use during training')
    
    # not used
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
