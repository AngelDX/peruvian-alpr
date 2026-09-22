#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evalua un modelo entrenado sobre el GOLD (o cualquier TSV) y guarda metricas
unificadas (exact-match, CER), con opcion de aplicar el corrector por posicion.

Modelo entrenado:
  python evaluate.py --ckpt ckpt/lprnet.pt --labels data/rec_labels_gold.tsv --name LPRNet
  python evaluate.py --ckpt ckpt/lprnet.pt --labels data/rec_labels_gold.tsv --name "LPRNet+corr" --corrector

Predicciones externas (p.ej. PaddleOCR) desde un TSV ya calculado:
  python evaluate.py --pred-tsv ../dataset_nuevo_validacion/resultados_verificacion.tsv \
        --pred-col placa_ocr --truth-col placa_correcta --name PaddleOCR
"""
import os, csv, json, argparse
from bench_utils import ctc_greedy_decode, metrics, norm_plate
from corrector import correct


def eval_model(ckpt, labels, use_corrector, device):
    import torch
    from torch.utils.data import DataLoader
    from datasets import PlateDataset, collate_ctc
    from models import build_model
    ck = torch.load(ckpt, map_location=device)
    model = build_model(ck["model"], ck["num_classes"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    H, W = ck["input_size"]
    dl = DataLoader(PlateDataset(labels, H, W), 256, shuffle=False, num_workers=4, collate_fn=collate_ctc)
    preds, truths = [], []
    with torch.no_grad():
        for imgs, _, _, plates in dl:
            idx = model(imgs.to(device)).cpu().argmax(2).permute(1, 0)
            for r, t in zip(idx, plates):
                p = ctc_greedy_decode(r.tolist())
                preds.append(correct(p) if use_corrector else p); truths.append(t)
    return preds, truths


def eval_pred_tsv(path, pred_col, truth_col, use_corrector):
    preds, truths = [], []
    with open(path, encoding="utf-8-sig") as f:
        sample = f.readline(); f.seek(0)
        delim = "\t" if "\t" in sample else (";" if ";" in sample else ",")
        for row in csv.DictReader(f, delimiter=delim):
            t = norm_plate(row[truth_col])
            if not t:
                continue
            p = norm_plate(row[pred_col])
            preds.append(correct(p) if use_corrector else p); truths.append(t)
    return preds, truths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--labels")
    ap.add_argument("--pred-tsv"); ap.add_argument("--pred-col", default="pred"); ap.add_argument("--truth-col", default="truth")
    ap.add_argument("--corrector", action="store_true")
    ap.add_argument("--name", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.ckpt:
        if a.device == "cpu":
            try:
                import torch; a.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass
        preds, truths = eval_model(a.ckpt, a.labels, a.corrector, a.device)
    elif a.pred_tsv:
        preds, truths = eval_pred_tsv(a.pred_tsv, a.pred_col, a.truth_col, a.corrector)
    else:
        raise SystemExit("usa --ckpt (modelo) o --pred-tsv (predicciones externas)")
    m = metrics(preds, truths); m["name"] = a.name; m["corrector"] = bool(a.corrector)
    json.dump(m, open(os.path.join(a.out, a.name.replace(" ", "_").replace("+", "p") + ".json"), "w"),
              ensure_ascii=False, indent=2)
    print("%-18s  exact=%.4f  CER=%.4f  (n=%d)" % (a.name, m["exact_match"], m["cer"], m["n"]))
