import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from model import EnhancedUNet

# 配置参数
TEST_VIDEO_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/validVid/"
MASK_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/dataPrep/json_et_png/maskVid"
OUTPUT_DIR = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/infRes/unet/UNet_100_1e-5_ReduceLRonPlateau"
CHECKPOINT_PATH = "/media/ubuntu-user/KESU/Intelligent_Medical_System/System/revised1/pth/UNet_100_1e-5_ReduceLRonPlateau.pth"
CLASS_SUBDIRS = ['Healthy', 'Polyp', 'GRED']
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据预处理
transform = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def process_video(input_path, output_path, model):
    """处理单个视频（包含真实标注叠加）"""
    # 初始化视频读取器
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 查找对应的真实标注视频
    video_name = os.path.basename(input_path)
    mask_path = None
    for cls in CLASS_SUBDIRS:
        possible_path = os.path.join(MASK_DIR, cls, video_name)
        if os.path.exists(possible_path):
            mask_path = possible_path
            break

    # 初始化标注视频读取器
    mask_cap = cv2.VideoCapture(mask_path) if mask_path else None

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理与模型推理
            orig_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = transform(image=orig_image)
            img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

            # 获取预测结果
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # 创建可视化画布
            vis_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            # 绘制预测轮廓（绿色）
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 3)

            # 绘制真实标注（如果存在）
            if mask_cap and mask_cap.isOpened():
                m_ret, mask_frame = mask_cap.read()
                if m_ret:
                    # 处理标注视频帧
                    gray_mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    gray_mask = cv2.resize(gray_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    _, gt_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

                    # 绘制真实轮廓（红色）
                    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_image, gt_contours, -1, (0, 0, 255), 2)

            # 写入输出视频
            out.write(vis_image)

    # 释放资源
    cap.release()
    if mask_cap: mask_cap.release()
    out.release()
    print(f"已保存结果视频: {output_path}")


def main():
    # 初始化模型
    model = EnhancedUNet(in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 遍历测试目录
    for root, dirs, files in os.walk(TEST_VIDEO_DIR):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                input_path = os.path.join(root, file)

                # 保持目录结构并强制使用.mp4扩展名
                rel_path = os.path.relpath(input_path, TEST_VIDEO_DIR)
                # 移除原扩展名并添加.mp4
                rel_path = os.path.splitext(rel_path)[0] + '.mp4'
                output_path = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 处理视频
                process_video(input_path, output_path, model)


if __name__ == "__main__":
    main()