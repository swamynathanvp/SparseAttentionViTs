from __future__ import print_function
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
from vit_pytorch import ViT
from models.big_bird_vit import BigBirdViT
import argparse
from torch.utils.tensorboard import SummaryWriter

# Argument Parser
parser = argparse.ArgumentParser('ViT Args', add_help=False)
parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--image_size', default=368, type=int)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--vit_arch', default="OriginalViT", type=str, help="OriginalViT, BigBirdViT")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.7, type=float)
parser.add_argument('--resume', default=False)
parser.add_argument('--tensorboard_dir', default="tensorboard")
parser.add_argument('--training_dir', default="training")
parser.add_argument('--resume_ckpt')

args = parser.parse_args()

def main(args):
    BASE_CKPT_PATH = r"C:\Users\vpswa\Downloads\SparseAttentionViTs-main\ckpt"
    MODEL_CKPT_PATH = os.path.join(BASE_CKPT_PATH, args.vit_arch)
    BASE_TENSORBOARD_PATH = r"C:\Users\vpswa\Downloads\SparseAttentionViTs-main\tensorboard"
    BASE_LOG_NAME = f"{args.image_size}_p{args.patch_size}"
    CUSTOM_LOG_NAME = "_BB_Global-200epochs_1e-3"
    RUN_LOG_NAME = BASE_LOG_NAME + CUSTOM_LOG_NAME
    RUN_CKPT_SAVE_PATH = os.path.join(BASE_CKPT_PATH, RUN_LOG_NAME)
    RUN_TENSORBOARD_PATH = os.path.join(BASE_TENSORBOARD_PATH, RUN_LOG_NAME)
    RESUME = args.resume
    resume_ckpt_path = args.resume_ckpt if args.resume_ckpt else ""

    os.makedirs(RUN_CKPT_SAVE_PATH, exist_ok=True)
    os.makedirs(RUN_TENSORBOARD_PATH, exist_ok=True)
    writer = SummaryWriter(RUN_TENSORBOARD_PATH)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    print(f"Using device: {DEVICE}")

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(root=r"C:\Users\vpswa\Downloads\SparseAttentionViTs-main\imagenette2\train", transform=transform_train)
    test_set = datasets.ImageFolder(root=r"C:\Users\vpswa\Downloads\SparseAttentionViTs-main\imagenette2\val", transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # Model Selection
    if args.vit_arch == "OriginalViT":
        model = ViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=10,
            dim=512,
            depth=3,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            pool='mean'
        )
    elif args.vit_arch == "BigBirdViT":
        attentions_to_use = ["Global","Window","Random"]
        model = BigBirdViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=10,
            dim=512,
            depth=3,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            attention_to_use=attentions_to_use,
            pool='mean'
        )
    else:
        print("Error: Unknown Model", args.vit_arch)
        exit()

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True

    model = model.to(DEVICE)

    if RESUME and resume_ckpt_path:
        model.load_state_dict(torch.load(resume_ckpt_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', verbose=True)

    best_val_acc = 0.0
    print("Training", args.vit_arch)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()
        for data, label in tqdm(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                val_output = model(data)
                val_loss = criterion(val_output, label)
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

        if best_val_acc < epoch_val_accuracy:
            torch.save(model.state_dict(), os.path.join(RUN_CKPT_SAVE_PATH, "best_ckpt.pt"))
            with open(os.path.join(RUN_CKPT_SAVE_PATH, "meta_json.json"), "w") as fp:
                json.dump({'Epoch': epoch+1, 'Val_acc': epoch_val_accuracy.item()}, fp)
            best_val_acc = epoch_val_accuracy

        writer.add_scalar('Train loss', epoch_loss, epoch)
        writer.add_scalar('Train acc', epoch_accuracy, epoch)
        writer.add_scalar('Val loss', epoch_val_loss, epoch)
        writer.add_scalar('Val acc', epoch_val_accuracy, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(epoch_val_loss)

    torch.save(model.state_dict(), os.path.join(RUN_CKPT_SAVE_PATH, f"best_ckpt_{epoch}.pt"))

if __name__ == '__main__':
    main(args)

# CLassic: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py
# BigBird: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py --vit_arch BigBirdViT
# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py --vit_arch BigBirdViT --patch_size=8 --image_size=184



#python train_vit.py --vit_arch BigBirdViT --patch_size=8 --image_size=384



# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --master_port 47770 train_vit.py --vit_arch OriginalViT --patch_size=8 --image_size=384
# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --master_port 47770 train_vit.py --vit_arch BigBirdViT --patch_size=8 --image_size=384