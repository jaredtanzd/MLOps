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

def train_net(net, device, epochs: int = 5, batch_size: int = 1, learning_rate: float = 1e-5, img_scale: float = 1.0, amp: bool = False):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale)
    n_train = len(dataset)
    n_val = len(dataset_val)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_scale=img_scale, amp=amp))

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    # Clear previous summary and logs
    summary_file_path = Path('summary.txt')
    training_log_path = Path('training_logs.txt')
    summary_file_path.unlink(missing_ok=True)
    training_log_path.unlink(missing_ok=True)

    write_summary("Epoch", "Learning Rate", "Step", "Train Loss", "Validation Dice", file_path=summary_file_path)  # Header

    global_step = 0
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Optional: Log training details for each batch
                log_training_info(f"Epoch {epoch}, Batch Loss: {loss.item()}", log_path=training_log_path)

            val_score = evaluate(net, val_loader, device)
            scheduler.step(val_score)

            write_summary(epoch, optimizer.param_groups[0]['lr'], global_step, epoch_loss / len(train_loader), val_score, file_path=summary_file_path)

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), dir_checkpoint / f'checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')

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
