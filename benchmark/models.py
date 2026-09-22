# -*- coding: utf-8 -*-
"""LPRNet y CRNN (salida CTC (T, B, num_classes)). Placas peruanas de 6 chars."""
import torch, torch.nn as nn


class LPRNet(nn.Module):
    """LPRNet compacto. Entrada 3xHxW -> secuencia T = W//4.

    A 32x100 (resolucion unificada del estudio) da T=25; a 24x94, T=23.
    El colapso de altura es adaptativo: a H=24 equivale exactamente al
    MaxPool2d((3,1),(3,1)) original, de modo que los checkpoints antiguos
    siguen dando las mismas predicciones.
    """
    input_size = (32, 100)
    def __init__(self, num_classes):
        super().__init__()
        def cbr(i, o, k=3, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, 1, p), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            cbr(3, 64), nn.MaxPool2d(2, 2),                 # H/2,  W/2
            cbr(64, 128), nn.MaxPool2d(2, 2),               # H/4,  W/4
            cbr(128, 256), nn.MaxPool2d((2, 1), (2, 1)),    # H/8,  W/4
            cbr(256, 256), nn.AdaptiveMaxPool2d((1, None)),  # 1,    W/4
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(256, num_classes, 1)    # (B,C,1,23)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.classifier(x)              # (B, C, 1, W)
        x = x.squeeze(2)                    # (B, C, W)
        x = x.permute(2, 0, 1)             # (T=W, B, C)
        return x


class CRNN(nn.Module):
    """CRNN clasico (CNN + BiLSTM). Entrada 3x32x100 -> secuencia T=24."""
    input_size = (32, 100)
    def __init__(self, num_classes, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),          # 16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),        # 8x25
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 4x25
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 2x25
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),      # 1x24
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)                     # (B, 512, 1, W)
        b, c, h, w = x.size()
        assert h == 1, "altura debe colapsar a 1 (revisar input_size)"
        x = x.squeeze(2).permute(2, 0, 1)  # (T=W, B, 512)
        x, _ = self.rnn(x)
        x = self.fc(x)                     # (T, B, C)
        return x


def build_model(name, num_classes):
    name = name.lower()
    if name == "lprnet":
        return LPRNet(num_classes)
    if name == "crnn":
        return CRNN(num_classes)
    raise ValueError("modelo desconocido: %s (usa lprnet o crnn)" % name)
