# sam_model.py (modified)
import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class SAMSeg(nn.Module):
    def __init__(self, num_classes=5, checkpoint="sam_vit_b_01ec64.pth"):
        super().__init__()

        # Initialize SAM model
        self.sam = sam_model_registry['vit_b'](
            checkpoint='/media/ubuntu-user/KESU/Intelligent_Medical_System/System/pth/sam_vit_b_01ec64.pth'
        )

        # Freeze image encoder except last 6 blocks
        for name, param in self.sam.named_parameters():
            if 'image_encoder' in name:
                if 'blocks.6' in name or 'blocks.7' in name or 'blocks.8' in name or 'blocks.9' in name or 'blocks.10' in name or 'blocks.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Adjusted Decoder (SAM's image_encoder output is [batch_size, 256, 64, 64])
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, 1, 1)
        )

        # Classification head (using SAM's image embedding)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Better for variable spatial sizes
            nn.Flatten(),
            nn.Linear(256, 512),  # 256 matches SAM's encoder output channels
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get SAM image embeddings (output shape: [batch_size, 256, 64, 64])
        image_embedding = self.sam.image_encoder(x)

        # Segmentation decoder
        seg_features = self.decoder(image_embedding)  # Output: [batch_size, 1, 1024, 1024]

        # Classification prediction
        cls_pred = self.cls_head(image_embedding)

        return seg_features, cls_pred