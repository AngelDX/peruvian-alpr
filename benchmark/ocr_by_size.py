#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precision de OCR en funcion del tamano de placa (px) sobre el gold (782).

Corre LPRNet y CRNN por-crop, toma PaddleOCR del TSV verificado, mide exact-match
por rango de AREA de la placa y guarda size_analysis.json para el dashboard.
"""
import os, csv, json, cv2, numpy as np, torch
from bench_utils import ctc_greedy_decode, norm_plate
from corrector import correct
from models import build_model

GOLD = "data/rec_labels_gold.tsv"
PADDLE = "../dataset_nuevo_validacion/resultados_verificacion.tsv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_gold():
    rows = []
    for l in open(GOLD, encoding="utf-8").read().splitlines():
        if not l: continue
        path, truth = l.split("\t")
        rows.append((path, truth))
    return rows


@torch.no_grad()
def infer(ckpt, rows):
    ck = torch.load(ckpt, map_location=DEV)
    model = build_model(ck["model"], ck["num_classes"]).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    H, W = ck["input_size"]
    preds = []
    B = 128; batch = []; idx = []
    def flush():
        if not batch: return
        x = torch.stack(batch).to(DEV)
        out = model(x).cpu().argmax(2).permute(1, 0)
        for r in out:
            preds.append(ctc_greedy_decode(r.tolist()))
        batch.clear()
    for path, _ in rows:
        im = cv2.imread(path)
        if im is None: im = np.zeros((H, W, 3), np.uint8)
        im = cv2.resize(im, (W, H)).astype(np.float32) / 127.5 - 1.0
        batch.append(torch.from_numpy(im.transpose(2, 0, 1)))
        if len(batch) >= B: flush()
    flush()
    return preds


def load_paddle():
    d = {}
    with open(PADDLE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fn = row["nombre_archivo"].strip().strip('"')
            d[fn] = (norm_plate(row["placa_ocr"]), norm_plate(row["placa_corregida"]), norm_plate(row["placa_correcta"]))
    return d


def main():
    rows = load_gold()
    sizes = []
    for path, _ in rows:
        im = cv2.imread(path); h, w = (im.shape[:2] if im is not None else (0, 0))
        sizes.append({"w": w, "h": h, "area": w * h})
    lp = infer("ckpt/lprnet.pt", rows)
    cr = infer("ckpt/crnn.pt", rows)
    paddle = load_paddle()

    per = []
    for (path, truth), sz, p_lp, p_cr in zip(rows, sizes, lp, cr):
        fn = os.path.basename(path)
        t = norm_plate(truth)
        pk = paddle.get(fn)
        p_raw = pk[0] if pk else ""
        p_corr = pk[1] if pk else ""
        per.append({
            "fn": fn, "w": sz["w"], "h": sz["h"], "area": sz["area"], "truth": t,
            "lprnet": int(norm_plate(p_lp) == t),
            "crnn": int(norm_plate(p_cr) == t),
            "paddle": int(p_corr == t),          # PaddleOCR + corrector (produccion)
            "paddle_raw": int(p_raw == t),
        })

    # bins por area (cuantiles ~quintiles redondeados)
    areas = np.array([p["area"] for p in per])
    edges = [int(areas.min()) - 1, 4000, 4700, 5300, 6200, int(areas.max()) + 1]
    labels = ["<4.0k", "4.0-4.7k", "4.7-5.3k", "5.3-6.2k", ">6.2k"]
    models = ["lprnet", "crnn", "paddle", "paddle_raw"]
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        grp = [p for p in per if lo <= p["area"] < hi]
        b = {"label": labels[i], "lo": max(lo, 0), "hi": hi, "n": len(grp)}
        for m in models:
            b[m] = round(sum(p[m] for p in grp) / len(grp), 4) if grp else None
        bins.append(b)

    # distribucion (histograma de area, 24 barras)
    hist, hedges = np.histogram(areas, bins=24)
    dist = {"counts": hist.tolist(), "edges": [int(x) for x in hedges]}

    # correlacion tamano-acierto (point-biserial ~ correlacion de Pearson con binaria)
    corr = {}
    for m in models:
        y = np.array([p[m] for p in per], dtype=float)
        corr[m] = round(float(np.corrcoef(areas, y)[0, 1]), 4)

    out = {
        "n": len(per), "bins": bins, "models": models, "dist": dist, "corr": corr,
        "overall": {m: round(sum(p[m] for p in per) / len(per), 4) for m in models},
        "area_stats": {"min": int(areas.min()), "median": int(np.median(areas)), "max": int(areas.max())},
        "w_range": [int(min(p["w"] for p in per)), int(max(p["w"] for p in per))],
        "h_range": [int(min(p["h"] for p in per)), int(max(p["h"] for p in per))],
    }
    json.dump(out, open("size_analysis.json", "w"), ensure_ascii=False, indent=2)
    # imprimir tabla
    print("AREA px2      n    LPRNet   CRNN   Paddle+corr  Paddle_raw")
    for b in bins:
        print("%-10s %4d   %5.1f%%  %5.1f%%   %5.1f%%     %5.1f%%" %
              (b["label"], b["n"], b["lprnet"]*100, b["crnn"]*100, b["paddle"]*100, b["paddle_raw"]*100))
    print("corr(area, acierto):", corr)


if __name__ == "__main__":
    main()
