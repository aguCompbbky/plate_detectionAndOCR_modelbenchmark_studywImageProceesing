"""
LPRNet Model Architecture
License Plate Recognition Network with small backbone + global context embedding
Input: (batch, 3, 24, 94) -> Output: (batch, num_classes, seq_len)
"""

import torch
import torch.nn as nn


CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U',
    'V', 'Y', 'Z'
]

CHARS_DICT = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 33 chars + 1 CTC blank


class SmallBasicBlock(nn.Module):
    """Small basic block with 1xN and Nx1 convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    """
    LPRNet: License Plate Recognition via Deep Neural Networks
    Adapted for Turkish license plates.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            # Stage 2
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/2

            # Stage 3
            SmallBasicBlock(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SmallBasicBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/4

            # Stage 4
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=1, padding=0),  # collapse H
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(256, num_classes, kernel_size=(1, 13), stride=1, padding=(0, 6)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

        # Global context embedding
        self.container = nn.Sequential(
            nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # x: (B, 3, 24, 94)
        features = self.backbone(x)  # (B, num_classes, 3, W)

        # Global context
        # features is (B, C, H, W) where H is currently 3 (because 24 -> 12 -> 6 -> 3)
        # We want to pool along H and W. But let's first simplify to (B, C, 1, W)
        features = torch.mean(features, dim=2, keepdim=True) # (B, C, 1, W)
        
        T = features.size(3)
        avg_pool = torch.mean(features, dim=3, keepdim=True).expand_as(features)  # (B, C, 1, W)
        max_pool = torch.max(features, dim=3, keepdim=True)[0].expand_as(features)  # (B, C, 1, W)

        # Concatenate features with global context
        combined = torch.cat([features, avg_pool, max_pool], dim=1)  # (B, C*3, 1, W)
        output = self.container(combined)  # (B, num_classes, 1, W)
        output = output.squeeze(2)  # (B, num_classes, W)

        # Log softmax for CTC
        logits = output.permute(2, 0, 1)  # (W, B, num_classes) for CTC
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return log_probs  # (T, B, C)


def build_lprnet(num_classes=NUM_CLASSES, pretrained_path=None):
    """Build LPRNet model, optionally loading pretrained weights."""
    model = LPRNet(num_classes=num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    return model
