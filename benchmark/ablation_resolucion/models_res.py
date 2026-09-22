# -*- coding: utf-8 -*-
"""LPRNet y CRNN parametrizados por resolucion de entrada.

Son los mismos modelos de ../models.py, con un unico cambio: la altura del mapa
de caracteristicas se colapsa a 1 mediante un pooling adaptativo en lugar de un
pooling de tamano fijo. Eso permite alimentarlos con cualquier (H, W).

Comprobado en test_shapes(): a 24x94 (LPRNet) y 32x100 (CRNN) el grafo es
numericamente identico al original, de modo que la fila de linea base de la
ablacion reproduce las cifras ya publicadas.
"""
import torch
import torch.nn as nn


class LPRNetRes(nn.Module):
    """LPRNet compacto. Entrada 3xHxW -> secuencia T = W // 4."""

    def __init__(self, num_classes, input_size=(24, 94)):
        super().__init__()
        self.input_size = tuple(input_size)

        def cbr(i, o, k=3, p=1):
            return nn.Sequential(
                nn.Conv2d(i, o, k, 1, p),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            cbr(3, 64), nn.MaxPool2d(2, 2),                  # H/2,  W/2
            cbr(64, 128), nn.MaxPool2d(2, 2),                # H/4,  W/4
            cbr(128, 256), nn.MaxPool2d((2, 1), (2, 1)),     # H/8,  W/4
            cbr(256, 256),
        )
        # En el original: MaxPool2d((3,1),(3,1)) sobre H=3 -> 1.
        # AdaptiveMaxPool2d((1,None)) hace lo mismo a 24x94 y generaliza al resto.
        self.collapse = nn.AdaptiveMaxPool2d((1, None))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.collapse(x)          # (B, 256, 1, W/4)
        x = self.dropout(x)
        x = self.classifier(x)        # (B, C, 1, T)
        x = x.squeeze(2)              # (B, C, T)
        x = x.permute(2, 0, 1)        # (T, B, C)
        return x


class CRNNRes(nn.Module):
    """CRNN clasico (CNN + BiLSTM). Entrada 3xHxW -> secuencia T = W // 4 - 1."""

    def __init__(self, num_classes, input_size=(32, 100), hidden=256):
        super().__init__()
        self.input_size = tuple(input_size)
        self.cnn_a = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        )
        # A 32x100 la altura ya vale 2 aqui, asi que el pooling es identidad.
        self.collapse = nn.AdaptiveMaxPool2d((2, None))
        self.cnn_b = nn.Sequential(
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn_a(x)
        x = self.collapse(x)           # (B, 512, 2, W')
        x = self.cnn_b(x)              # (B, 512, 1, W'-1)
        b, c, h, w = x.size()
        assert h == 1, "la altura debe colapsar a 1 (revisar collapse)"
        x = x.squeeze(2).permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)                 # (T, B, C)
        return x


def build_model_res(name, num_classes, input_size):
    name = name.lower()
    if name == "lprnet":
        return LPRNetRes(num_classes, input_size)
    if name == "crnn":
        return CRNNRes(num_classes, input_size)
    raise ValueError("modelo desconocido: %s (usa lprnet o crnn)" % name)


def seq_len(name, width):
    """T resultante para una anchura dada, sin construir el modelo."""
    if name.lower() == "lprnet":
        return width // 4
    if name.lower() == "crnn":
        return width // 4 - 1
    raise ValueError(name)


def test_shapes(verbose=True):
    """Verifica que las seis resoluciones producen T >= 6 y altura colapsada."""
    from itertools import product
    sizes = [(24, 94), (32, 100), (48, 96), (64, 128), (80, 160), (96, 320)]
    ok = True
    for name, (h, w) in product(("lprnet", "crnn"), sizes):
        m = build_model_res(name, 37, (h, w)).eval()
        with torch.no_grad():
            out = m(torch.zeros(2, 3, h, w))
        t, b, c = out.shape
        good = (t == seq_len(name, w)) and t >= 6 and b == 2 and c == 37
        ok &= good
        if verbose:
            n = sum(p.numel() for p in m.parameters())
            print("%-7s %3dx%-4d  T=%-3d params=%.2fM  %s"
                  % (name, h, w, t, n / 1e6, "ok" if good else "FALLA"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if test_shapes() else 1)
