import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from albumentations import Compose, Resize, Normalize, ToTensorV2
from sam_model import SAMSeg

# Configuration
TEST_VIDEO_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/validVid/"
OUTPUT_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/infRes/sam/SAMSeg_100_1e-4_111_ReduceLRonPlateau"
CHECKPOINT_PATH = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/pth/SAMSeg_100_1e-4_111_ReduceLRonPlateau.pth"
MASK_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataPrep/json_et_png/maskVid"
FRAME_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataPrep/json_et_png/frameVid"
IMG_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_SUBDIRS = ['Healthy', 'Polyp', 'GRED']

# Preprocessing transformations
transform = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load trained model
model = SAMSeg(num_classes=3)
try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit()
model.to(device)
model.eval()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each video
for video_path in glob.glob(os.path.join(TEST_VIDEO_DIR, '**', '*.avi'), recursive=True):
    # Create output path preserving directory structure
    rel_path = os.path.relpath(video_path, TEST_VIDEO_DIR)
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_name = os.path.basename(video_path)
    mask_reader = None

    # Search for corresponding mask
    mask_path = None
    for cls in CLASS_SUBDIRS:
        candidate = os.path.join(MASK_DIR, cls, video_name)
        if os.path.exists(candidate):
            mask_path = candidate
            break

    # Initialize mask reader
    if mask_path:
        mask_reader = cv2.VideoCapture(mask_path)
        if not mask_reader.isOpened():
            print(f"Error opening mask video: {mask_path}")
            mask_reader = None

    # Initialize video reader
    vr = cv2.VideoCapture(video_path)
    if not vr.isOpened():
        print(f"Error opening video: {video_path}")
        continue

    # Get video properties
    fps = vr.get(cv2.CAP_PROP_FPS)
    total_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (IMG_SIZE, IMG_SIZE), isColor=True)

    frame_count = 0
    while True:
        # Read input frame
        ret, frame = vr.read()
        if not ret:
            break

        # Convert color space and process
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transform(image=frame)
        img_tensor = transformed["image"].unsqueeze(0).to(device)

        # Model prediction with memory management
        with torch.no_grad():
            seg_pred, cls_pred = model(img_tensor)
            seg_pred = F.interpolate(seg_pred, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

        # Cleanup GPU memory
        del img_tensor
        torch.cuda.empty_cache()

        # Get predictions
        pred_class = torch.argmax(cls_pred, dim=1).item()
        prob_map = seg_pred.sigmoid().squeeze().cpu().numpy()

        # Prepare visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        denorm_img = (transformed["image"].to(device) * std + mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        denorm_img = (denorm_img * 255).astype(np.uint8)
        denorm_img_bgr = cv2.cvtColor(denorm_img, cv2.COLOR_RGB2BGR)

        # Create overlay
        overlay = denorm_img_bgr.copy()

        # Process prediction mask
        threshold = 0.3
        binary_mask = (prob_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

        # Process ground truth if available
        if mask_reader and mask_reader.isOpened():
            ret_mask, mask_frame = mask_reader.read()
            if ret_mask:
                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                mask_gray = cv2.resize(mask_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                _, gt_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, gt_contours, -1, (0, 0, 255), 3)
                cv2.putText(overlay, "Ground Truth", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Calculate Dice coefficient
                y_true = gt_mask.astype(np.float32) / 255.0
                y_pred = binary_mask.astype(np.float32)
                intersection = np.sum(y_true * y_pred)
                dice = (2. * intersection + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)
                cv2.putText(overlay, f"Dice: {dice:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add class prediction
        cls_prob = F.softmax(cls_pred, dim=1).squeeze().cpu().numpy()
        class_label = f"{CLASS_SUBDIRS[pred_class]} ({cls_prob[pred_class]:.2f})"
        cv2.putText(overlay, class_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(overlay, "Prediction", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write to output
        out.write(overlay)
        frame_count += 1

    # Release resources
    vr.release()
    if mask_reader:
        mask_reader.release()
    out.release()
    print(f"Processed {frame_count} frames: {video_path} -> {output_path}")

print("Inference completed!")