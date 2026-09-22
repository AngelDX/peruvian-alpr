#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exporta el LPRNet afinado a ONNX para la app Android y VERIFICA la paridad.

La app corre el OCR sobre ONNX Runtime, asi que el checkpoint de PyTorch hay que
convertirlo. Una exportacion que degrada en silencio es el peor resultado posible,
de modo que el script vuelve a medir el exact-match del modelo exportado sobre el
mismo gold y lo compara con el del checkpoint: si no coinciden, falla.

Se envuelve el modelo para que la salida sea (B, T, C) con softmax aplicado, igual
que el ONNX de PaddleOCR que ya consume PlateRecognizer.kt (evita tocar el decodificador).

Uso:
    python export_lprnet_onnx.py --ckpt ckpt/lprnet_ft_2lotes.pt
"""
import argparse, csv, sys
from pathlib import Path

import torch
import torch.nn as nn

from bench_utils import CHARS, BLANK, I2C, norm_plate
from models import build_model


class OcrWrapper(nn.Module):
    """LPRNet -> (B, T, C) con softmax, que es lo que espera el decodificador CTC de la app."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        logits = self.base(x)               # (T, B, C)
        probs = logits.permute(1, 0, 2)     # (B, T, C)
        return torch.softmax(probs, dim=-1)


def ctc_greedy(probs_bt_c):
    """Decodifica una secuencia (T, C) de probabilidades."""
    out, prev = [], -1
    for t in range(probs_bt_c.shape[0]):
        idx = int(probs_bt_c[t].argmax())
        if idx != BLANK and idx != prev:
            out.append(I2C.get(idx, ""))
        prev = idx
    return "".join(out)


def cargar_gold(tsv, limite=None):
    filas = []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                filas.append((row[0], norm_plate(row[1])))
    return filas[:limite] if limite else filas


def preparar_lote(rutas, H, W):
    """Calca EXACTAMENTE el preproceso de PlateDataset (datasets.py):
    cv2.imread -> BGR, cv2.resize (W,H) bilineal, /127.5 - 1.0 -> [-1,1], CHW.

    Los tres detalles importan y hay que replicarlos en Android: el orden de canal es
    BGR (no RGB) y el rango es [-1,1] (no [0,1]).
    """
    import cv2, numpy as np
    lote = []
    for r in rutas:
        img = cv2.imread(r)
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        img = cv2.resize(img, (W, H))
        img = img.astype(np.float32) / 127.5 - 1.0
        lote.append(img.transpose(2, 0, 1))
    return np.stack(lote)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/lprnet_ft_2lotes.pt")
    ap.add_argument("--gold", default="data/rec_labels_gold.tsv")
    ap.add_argument("--out", default="../app_android/app/src/main/assets/plate_ocr_lprnet.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort

    ck = torch.load(args.ckpt, map_location="cpu")
    H, W = ck["input_size"]
    print(f"Checkpoint: {args.ckpt}")
    print(f"  arquitectura={ck['model']}  entrada={H}x{W}  clases={ck['num_classes']}  "
          f"exact-match declarado={ck.get('val_exact', float('nan')):.4f}")
    if ck["num_classes"] != len(CHARS) + 1:
        sys.exit(f"num_classes={ck['num_classes']} no cuadra con el charset ({len(CHARS)}+blank)")

    base = build_model(ck["model"], ck["num_classes"])
    base.load_state_dict(ck["state_dict"])
    base.eval()
    modelo = OcrWrapper(base).eval()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, H, W)
    torch.onnx.export(
        modelo, dummy, str(out_path),
        input_names=["images"], output_names=["probs"],
        dynamic_axes={"images": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
        # Exportador clasico: la red es Conv/BN/ReLU/MaxPool/softmax, no necesita dynamo
        # (que ademas exigiria onnxscript como dependencia extra).
        dynamo=False,
    )
    print(f"\nExportado: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    # --- Verificacion de paridad sobre el gold completo
    filas = cargar_gold(args.gold)
    print(f"\nVerificando sobre {len(filas)} placas del gold ...")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    entrada = sess.get_inputs()[0].name

    ok_torch = ok_onnx = coinciden = 0
    max_diff = 0.0
    B = 64
    for i in range(0, len(filas), B):
        trozo = filas[i:i + B]
        x = preparar_lote([r for r, _ in trozo], H, W)
        with torch.no_grad():
            p_torch = modelo(torch.from_numpy(x)).numpy()
        p_onnx = sess.run(None, {entrada: x})[0]
        max_diff = max(max_diff, float(np.abs(p_torch - p_onnx).max()))
        for j, (_, verdad) in enumerate(trozo):
            t = ctc_greedy(p_torch[j])
            o = ctc_greedy(p_onnx[j])
            ok_torch += (t == verdad)
            ok_onnx += (o == verdad)
            coinciden += (t == o)

    n = len(filas)
    print(f"  exact-match PyTorch : {ok_torch}/{n} = {ok_torch/n:.4f}")
    print(f"  exact-match ONNX    : {ok_onnx}/{n} = {ok_onnx/n:.4f}")
    print(f"  predicciones identicas: {coinciden}/{n}")
    print(f"  diferencia numerica maxima: {max_diff:.2e}")

    if coinciden != n:
        sys.exit(f"\nFALLO: el ONNX difiere del checkpoint en {n - coinciden} placas")
    print("\nParidad exacta: el ONNX reproduce el checkpoint placa por placa.")

    dic = out_path.parent / "lprnet_chars.txt"
    dic.write_text("\n".join(CHARS) + "\n", encoding="utf-8")
    print(f"Charset ({len(CHARS)} simbolos, blank=0 implicito): {dic}")


if __name__ == "__main__":
    main()
