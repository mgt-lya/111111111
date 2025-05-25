# train_sam.py
import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as T
import albumentations as A
import torch.nn.functional as F
import decord
from sam_model import SAMSeg

# Config
IMG_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/pth"
FRAME_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataPrep/json_et_png/frameVid"
MASK_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataPrep/json_et_png/maskVid"
CLASS_NAMES = ['Healthy', 'Polyp', 'GRED']
LOG_FILE = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/SAMSeg_100_0.2_1e-4_111_ReduceLRonPlateau.txt"


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class MedicalDataset(Dataset):
    def __init__(self, frame_dir, mask_dir, transform=None):
        self.samples = []
        self.class_counts = np.zeros(len(CLASS_NAMES))
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
        self.transform = transform
        self.vr_cache = {}

        for class_name in CLASS_NAMES:
            class_idx = self.class_to_idx[class_name]
            frame_class_dir = os.path.join(frame_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)

            if not os.path.exists(frame_class_dir) or not os.path.exists(mask_class_dir):
                continue

            video_pairs = []
            for vid_name in os.listdir(frame_class_dir):
                frame_vid = os.path.join(frame_class_dir, vid_name)
                mask_vid = os.path.join(mask_class_dir, vid_name)

                if os.path.exists(mask_vid):
                    video_pairs.append((frame_vid, mask_vid))

            for f_vid, m_vid in video_pairs:
                vr = decord.VideoReader(f_vid)
                num_frames = len(vr)
                self.samples.extend([(f_vid, m_vid, i, class_idx) for i in range(num_frames)])
                self.class_counts[class_idx] += num_frames

        print(f"Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_path, m_path, frame_idx, cls_idx = self.samples[idx]

        if f_path not in self.vr_cache:
            self.vr_cache[f_path] = decord.VideoReader(f_path)
        frame = self.vr_cache[f_path][frame_idx].asnumpy()

        if m_path not in self.vr_cache:
            self.vr_cache[m_path] = decord.VideoReader(m_path)
        mask = self.vr_cache[m_path][frame_idx].asnumpy()
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=frame, mask=mask)
            frame = transformed["image"]
            mask = transformed["mask"]

        return frame, mask.unsqueeze(0), torch.tensor(cls_idx)


class MultiTaskLoss(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        self.class_weights = 1.0 / (torch.sqrt(torch.tensor(class_counts).float() + 1e-6))
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.focal_loss = FocalLoss(weight=self.class_weights, gamma=2)

    def forward(self, seg_pred, cls_pred, seg_target, cls_target):
        seg_pred = seg_pred.squeeze(1)
        seg_target = seg_target.squeeze(1).float()

        with torch.no_grad():
            edges = F.max_pool2d(seg_target, 3, 1, 1) - F.avg_pool2d(seg_target, 3, 1, 1)
            edge_weight = 1.0 + 2.0 * (edges > 0.1).float()

        sum_target = (seg_target * edge_weight).sum()
        sum_pred = (seg_pred.sigmoid() * edge_weight).sum()
        intersection = (seg_pred.sigmoid() * seg_target * edge_weight).sum()

        dice = (2. * intersection + 1e-6) / (sum_pred + sum_target + 1e-6)
        dice = torch.where(sum_target > 0, dice, 1.0 - (sum_pred > 0).float())
        dice_loss = 1 - dice.mean()

        pos_weight = torch.exp(-5 * seg_target.mean())
        focal_loss = F.binary_cross_entropy_with_logits(
            seg_pred, seg_target,
            reduction='mean',
            pos_weight=pos_weight + 1.0
        )

        cls_loss = self.focal_loss(cls_pred, cls_target)

        return 1.0 * dice_loss + 1.0 * focal_loss + 1.0 * cls_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    full_dataset = MedicalDataset(FRAME_DIR, MASK_DIR, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    class_weights = 1.0 / (full_dataset.class_counts + 1e-6)
    sample_weights = class_weights[[s[3] for s in full_dataset.samples]]
    train_sampler = WeightedRandomSampler(sample_weights[:len(train_set)], len(train_set))

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, drop_last=True)

    model = SAMSeg(num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = MultiTaskLoss(full_dataset.class_counts)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',        # Monitor IoU metric
        factor=0.5,        # Reduce LR by half
        patience=3,        # Wait 3 epochs without improvement
        verbose=True,      # Print update messages
        min_lr=1e-6        # Minimum learning rate
    )
    scaler = torch.amp.GradScaler()

    best_iou = 0.0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y, cls in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            x, y, cls = x.to(device), y.to(device), cls.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                seg_pred, cls_pred = model(x)
                loss = criterion(seg_pred, cls_pred, y, cls)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        tp_per_class = torch.zeros(len(CLASS_NAMES), device=device)
        fp_per_class = torch.zeros(len(CLASS_NAMES), device=device)
        fn_per_class = torch.zeros(len(CLASS_NAMES), device=device)
        tn_per_class = torch.zeros(len(CLASS_NAMES), device=device)
        cls_correct = torch.zeros(len(CLASS_NAMES), device=device)
        cls_total = torch.zeros(len(CLASS_NAMES), device=device)

        # Modify the validation loop's segmentation metrics calculation
        with torch.no_grad():
            for x, y, cls in val_loader:
                x, y, cls = x.to(device), y.to(device), cls.to(device)
                seg_pred, cls_pred = model(x)
                val_loss += criterion(seg_pred, cls_pred, y, cls).item()

                seg_pred = F.interpolate(seg_pred, size=(1024, 1024), mode='bilinear')
                pred_mask = (seg_pred.sigmoid() > 0.5).float()  # [B, 1, H, W]

                # Classification accuracy
                _, predicted_cls = torch.max(cls_pred, 1)
                for c in range(len(CLASS_NAMES)):
                    mask = (cls == c)
                    cls_correct[c] += (predicted_cls[mask] == c).sum()
                    cls_total[c] += mask.sum()

                # Segmentation metrics - modified for single-channel output
                for i in range(x.size(0)):  # Iterate through each sample in batch
                    c = cls[i].item()  # Get true class index for this sample
                    y_i = y[i].unsqueeze(0)  # [1, 1, H, W]
                    pred_i = pred_mask[i].unsqueeze(0)  # [1, 1, H, W]

                    tp = torch.sum(pred_i * y_i)
                    fp = torch.sum(pred_i * (1 - y_i))
                    fn = torch.sum((1 - pred_i) * y_i)
                    tn = torch.sum((1 - pred_i) * (1 - y_i))

                    tp_per_class[c] += tp
                    fp_per_class[c] += fp
                    fn_per_class[c] += fn
                    tn_per_class[c] += tn

        # Calculate metrics
        class_metrics = []
        for c in range(len(CLASS_NAMES)):
            tp = tp_per_class[c].item()
            fp = fp_per_class[c].item()
            fn = fn_per_class[c].item()
            tn = tn_per_class[c].item()

            iou = tp / (tp + fp + fn + 1e-7)
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            class_metrics.append({
                'iou': iou,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            })

        mean_iou = np.mean([m['iou'] for m in class_metrics])
        mean_accuracy = np.mean([m['accuracy'] for m in class_metrics])
        mean_precision = np.mean([m['precision'] for m in class_metrics])
        mean_recall = np.mean([m['recall'] for m in class_metrics])
        cls_accuracy = (cls_correct / (cls_total + 1e-7)).cpu().numpy()
        mean_cls_accuracy = np.mean(cls_accuracy)
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)  # 添加验证损失计算
        scheduler.step(mean_iou)

        # Logging
        log_str = f"\nEpoch {epoch + 1}/{EPOCHS}\n"
        log_str += f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        log_str += f"Val mIoU: {mean_iou:.4f} | Val Accuracy: {mean_accuracy:.4f}\n"
        log_str += f"Val Precision: {mean_precision:.4f} | Val Recall: {mean_recall:.4f}\n"
        log_str += f"Val Cls Acc: {mean_cls_accuracy:.4f}\n\nClass-wise Metrics:\n"

        for c, name in enumerate(CLASS_NAMES):
            log_str += f"{name}:\n"
            log_str += f"  IoU: {class_metrics[c]['iou']:.4f} | Accuracy: {class_metrics[c]['accuracy']:.4f}\n"
            log_str += f"  Precision: {class_metrics[c]['precision']:.4f} | Recall: {class_metrics[c]['recall']:.4f}\n"
            log_str += f"  Cls Acc: {cls_accuracy[c]:.4f} | Samples: {cls_total[c].item():.0f}\n"

        print(log_str)
        with open(LOG_FILE, 'a') as f:
            f.write(log_str)

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "SAMSeg_100_1e-4_0.2_111_ReduceLRonPlateau.pth"))
            best_msg = f"New best model saved with mIoU: {mean_iou:.4f}\n"
            print(best_msg)
            with open(LOG_FILE, 'a') as f:
                f.write(best_msg)


if __name__ == "__main__":
    main()