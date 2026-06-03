"""
model.py — CRNN architecture: VGG-style CNN + BiLSTM + CTC head.
"""
import torch
import torch.nn as nn

from config import IMG_H, NUM_CLASSES


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for sequence recognition.

    Input  : [B, 1, H, W]   (grayscale, H=32, W=128)
    Output : [T, B, num_classes]   (CTC log-softmax)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 256):
        super().__init__()

        # ── CNN feature extractor (VGG-style) ──────────────────────────────
        # Height pooling schedule: pool on H at layers 1,2; thereafter only W
        # After 2 H-max-pools: H = 32 → 8  → target 1 (handled by AdaptivePool)
        self.cnn = nn.Sequential(
            # Block 1
            _ConvBnRelu(1,   64),
            nn.MaxPool2d(2, 2),                       # H: 32→16  W: 128→64

            # Block 2
            _ConvBnRelu(64,  128),
            nn.MaxPool2d(2, 2),                       # H: 16→8   W: 64→32

            # Block 3
            _ConvBnRelu(128, 256),
            _ConvBnRelu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),             # H: 8→4    W: 32

            # Block 4
            _ConvBnRelu(256, 512),
            _ConvBnRelu(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),             # H: 4→2    W: 32

            # Block 5
            _ConvBnRelu(512, 512, k=2, s=1, p=0),    # H: 2→1    W: 31
        )
        # After CNN: [B, 512, 1, ~31]

        # ── Map-to-Sequence via AdaptiveAvgPool on H ───────────────────────
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))   # [B, 512, 1, W']

        # ── BiLSTM ────────────────────────────────────────────────────────
        self.rnn = nn.Sequential(
            _BiLSTMLayer(512,        rnn_hidden),
            _BiLSTMLayer(rnn_hidden, rnn_hidden),
        )

        # ── CTC classification head ────────────────────────────────────────
        self.classifier = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        feat = self.cnn(x)               # [B, 512, 1, W']
        feat = self.pool_h(feat)         # [B, 512, 1, W']
        feat = feat.squeeze(2)           # [B, 512, W']
        feat = feat.permute(2, 0, 1)    # [W', B, 512]  = [T, B, C]
        out  = self.rnn(feat)            # [T, B, rnn_hidden]
        out  = self.classifier(out)      # [T, B, num_classes]
        out  = out.log_softmax(2)
        return out                        # [T, B, num_classes]


class _BiLSTMLayer(nn.Module):
    def __init__(self, in_size: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        out, _ = self.lstm(x)    # [T, B, hidden*2]
        out = self.linear(out)   # [T, B, hidden]
        return out


# ─── Convenience ──────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, rnn_hidden: int = 256) -> CRNN:
    return CRNN(num_classes=num_classes, rnn_hidden=rnn_hidden)
