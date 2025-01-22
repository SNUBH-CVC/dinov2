import argparse
import re
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.utils.utils import load_pretrained_model_wo_blocks


class ViewClassificationDataset(Dataset):
    classes_str_to_int = {
        "AP": 0,
        "AP_CAUD": 1,
        "AP_CRAN": 2,
        "LAO": 3,
        "LAO_CAUD": 4,
        "LAO_CRAN": 5,
        "RAO": 6,
        "RAO_CAUD": 7,
        "RAO_CRAN": 8,
    }

    def __init__(self, data_root: str, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.data = self.load_data_list()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        img_path = item['img_path']
        label = item['gt_label']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def load_data_list(self):
        data_root = Path(self.data_root)
        split_dir_path = data_root / self.split

        data_list = []
        for k, v in self.classes_str_to_int.items():
            label = v
            img_dir_path = split_dir_path / k
            for img_path in img_dir_path.glob("*.png"):
                data_list.append({"img_path": str(img_path), "gt_label": label})
        return data_list



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc, all_preds, all_labels


def plot_confusion_matrix(true_labels, pred_labels, classes, save_path=None):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--data_root", type=str, default="./data/cag_view_classification")
    parser.add_argument("--ckpt_path", type=str, default="./pretrain/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./eval_outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = 518
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([123.675, 116.28, 103.53]) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375]) / 255.0
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create datasets
    train_dataset = ViewClassificationDataset(
        data_root=args.data_root,
        split="train",
        transform=transform
    )
    val_dataset = ViewClassificationDataset(
        data_root=args.data_root,
        split="val",
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Modify model for classification
    num_classes = len(ViewClassificationDataset.classes_str_to_int)
    if args.model == "dinov2":
        model = DinoVisionTransformer(
            img_size=img_size,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        if args.pretrained:
            print("Loading pretrained model")
            load_pretrained_model_wo_blocks(model, torch.load(args.ckpt_path))
        model.head = nn.Linear(model.embed_dim, num_classes)
    elif args.model == "resnet":
        print("Loading pretrained model")
        model = resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)   
    else:
        raise ValueError(f"Model {args.model} not supported")

    model = model.to(args.device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, args.device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            
            # Generate classification report
            class_names = list(ViewClassificationDataset.classes_str_to_int.keys())
            report = classification_report(
                val_labels,
                val_preds,
                target_names=class_names,
                digits=3
            )
            print("\nClassification Report:")
            print(report)
            
            # Plot and save confusion matrix
            plot_confusion_matrix(
                val_labels,
                val_preds,
                class_names,
                save_path=save_dir / 'confusion_matrix.png'
            )
    
    # Plot and save training metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path=save_dir / 'training_metrics.png')
    
    print(f'\nBest validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()