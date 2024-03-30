import argparse
import logging
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
# import wandb

from evaluate import evaluate
from unet.unet_model import UNet
from utils.data_loading import BasicDataset

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


def write_summary(epoch, lr, step, train_loss, val_dice, file_path='summary.txt'):
    # Convert to scalars if any value is a tensor
    train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
    val_dice = val_dice.item() if isinstance(val_dice, torch.Tensor) else val_dice

    with open(str(file_path), 'a') as file:  # Ensure file_path is a string
        file.write(f"{epoch}, {lr}, {step}, {train_loss}, {val_dice}\n")


def log_training_info(info, log_path='training_logs.txt'):
    with open(log_path, 'a') as file:
        file.write(f"{info}\n")

def freeze_model_layers(model, freeze_until_fn):
    for name, child in model.named_children():
        if not freeze_until_fn(name):
            for param in child.parameters():
                param.requires_grad = False

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Initialize Weights & Biases
    # if args.wandb:
    #     wandb.init(project="unet-finetuning", config=args)

    net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
    
    # Load the model if a path is provided
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device)
    
    # Load datasets
    train_dataset = BasicDataset(Path(args.dir_img), Path(args.dir_mask))
    val_dataset = BasicDataset(Path(args.dir_img_val), Path(args.dir_mask_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # Optimization and loss
    optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Check if the directory for the checkpoint exists; if not, create it
    checkpoint_directory = 'checkpoints_fine_tuning'
    os.makedirs(checkpoint_directory, exist_ok=True)  

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.001, path='checkpoints_fine_tuning/early_stopping_checkpoint.pth')

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs, true_masks = batch['image'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation part
        val_dice = evaluate(net, val_loader, device)
        
        # Early Stopping check
        early_stopping(val_dice, net)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break

        logging.info(f'Epoch {epoch} finished! Train Loss: {epoch_loss / len(train_loader):.4f}, Val Dice: {val_dice:.4f}')
        
        # if args.wandb:
        #     wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_loader), "val_dice": val_dice})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune U-Net model on images and target masks")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training and evaluation.')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--dir-img', type=str, required=True, help='Path to the directory containing training images.')
    parser.add_argument('--dir-mask', type=str, required=True, help='Path to the directory containing training mask images.')
    parser.add_argument('--dir-img-val', type=str, required=True, help='Path to the directory containing validation images.')
    parser.add_argument('--dir-mask-val', type=str, required=True, help='Path to the directory containing validation mask images.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use for training, if available.')
    parser.add_argument('--bilinear', action='store_true', help='Flag to use bilinear upsampling. If not set, transposed convolutions are used.')
    parser.add_argument('--load', type=str, help='Path to a .pth file from which to load a pretrained model.')
    # parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging training and validation metrics.')

    args = parser.parse_args()
    main(args)
