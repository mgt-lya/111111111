# train_model.py (Enhanced Version)
import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A
from model import EnhancedUNet  # Modified model architecture



# Configuration Parameters
IMG_SIZE = 512  # Increased input resolution
BATCH_SIZE = 2  # Optimized batch size
EPOCHS = 100
LEARNING_RATE = 1e-5  # Higher initial learning rate
PATIENCE = 10  # For early stopping
CHECKPOINT_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/pth"
FRAME_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataUnet/frameVid"
MASK_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataUnet/maskVid"
LOG_FILE = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/UNet_50_1e-3_512_ReduceLRonPlateau.txt"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class MedicalDataset(Dataset):
    def __init__(self, frame_dir, mask_dir, transform=None):
        self.frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.avi")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.avi")))
        self.transform = transform
        self.samples = []

        # Verify alignment and count frames
        for f_path, m_path in zip(self.frame_files, self.mask_files):
            cap = cv2.VideoCapture(f_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.samples.extend([(f_path, m_path, i) for i in range(frame_count)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_path, m_path, frame_idx = self.samples[idx]

        # Load frame with error handling
        cap = cv2.VideoCapture(f_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {f_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        # Load mask with validation
        cap = cv2.VideoCapture(m_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, mask = cap.read()
        if not ret:
            raise ValueError(f"Failed to read mask {frame_idx} from {m_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cap.release()

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=frame, mask=mask)
            frame = transformed["image"]
            mask = transformed["mask"]

        # Ensure valid mask values
        mask = (mask > 127).float()
        mask = mask.unsqueeze(0)  # 添加通道维度 [1, H, W]

        return frame, mask


class EnhancedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, gamma=2):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma  # 增加Focal Loss的gamma参数

    def forward(self, inputs, targets):
        # Ensure targets have correct dimensions
        if targets.shape[1] != 1:
            targets = targets[:, :1, :, :]

        # Compute probabilities for Dice Loss
        probs = torch.sigmoid(inputs)

        # Dice Loss
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)

        # Focal Loss (use raw logits)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal_loss = ((1 - p_t) ** self.gamma * bce).mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enhanced Data Augmentation
    transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ElasticTransform(
            sigma=15,  # 对应约3mm组织变形 (内镜FOV通常为50-100mm)
            alpha=30,  # 基于胃壁蠕动速度(2-5mm/s)
            p=0.5
        ),
        A.OneOf([
            A.CoarseDropout(  # 模拟液体遮挡
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=180,  # 黄色液体特征值
                p=0.5
            ),
        ], p=0.7),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ], is_check_shapes=False)

    # Create dataset with validation
    full_dataset = MedicalDataset(FRAME_DIR, MASK_DIR, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Use weighted sampler for class imbalance
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            pin_memory=True, num_workers=2)

    # Initialize model
    model = EnhancedUNet(in_channels=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # 使用更稳定的AdamW

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=15,  # Reset every 15 epochs
    #     T_mult=2,  # Double cycle length each time
    #     eta_min=1e-6,  # Minimum learning rate
    #     last_epoch=-1
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',        # Monitor IoU metric
        factor=0.5,        # Reduce LR by half
        patience=3,        # Wait 3 epochs without improvement
        verbose=True,      # Print update messages
        min_lr=1e-6        # Minimum learning rate
    )
    # total_steps = EPOCHS * len(train_loader)
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LEARNING_RATE,
    #     total_steps=total_steps,
    #     pct_start=0.3,
    #     anneal_strategy='cos',
    #     cycle_momentum=False  # Must disable for AdamW
    # )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    criterion = EnhancedLoss(dice_weight=1.0, focal_weight=0.5)
    early_stopping_counter = 0
    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Validation Phase
        model.eval()
        val_loss = 0.0
        total_intersection = 0
        total_union = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

                pred_mask = (torch.sigmoid(pred) > 0.5)  # 保持布尔型
                y_mask = (y > 0.5)

                total_intersection += (pred_mask & y_mask).sum().item()
                total_union += (pred_mask | y_mask).sum().item()

        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        iou = total_intersection / (total_union + 1e-7)

        # Update learning rate
        scheduler.step(iou)

        # document
        log_str = f"\nEpoch {epoch + 1}/{EPOCHS}\n"
        log_str += f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        log_str += f"Val mIoU: {iou:.4f}\n"

        # 打印并写入日志文件
        print(log_str)
        with open(LOG_FILE, 'a') as f:
            f.write(log_str)

        #print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            # Remove previous best model if exists
            prev_best = glob.glob(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            for pb in prev_best:
                os.remove(pb)
            # Save new best model
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "UNet_50_1e-3_512_ReduceLRonPlateau.pth"))
            #print(f"New best model saved with IoU: {best_iou:.4f}")

            best_msg = f"New best model saved with mIoU: {best_iou:.4f}\n"
            print(best_msg)
            with open(LOG_FILE, 'a') as f:
                f.write(best_msg)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
