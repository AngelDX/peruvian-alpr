#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selección activa de crops a etiquetar (verdad-terreno gold ampliada).

Corre LPRNet y CRNN sobre un pool de candidatos y prioriza los casos más
informativos para etiquetar manualmente (verificando con SUNARP):
  Tier A: LPRNet != CRNN            (modelos discrepan -> máxima incertidumbre)
  Tier B: ambos coinciden pero != silver (posible error del OCR/corrector)
  Tier C: placas pequeñas (área < umbral)  (régimen difícil de cámara a distancia)
Deduplica por placa (silver) y estratifica día/noche. Salida: active_selection.csv
Uso: python select_active.py --pool 120000 --n 5000
"""
import os, re, csv, random, argparse, datetime
import numpy as np, torch, cv2
from bench_utils import ctc_greedy_decode, norm_plate
from models import build_model

RES = re.compile(r"\((\d+),\s*(\d+),\s*\d+\)|(\d+)x(\d+)")
def area_of(path):
    m = RES.search(os.path.basename(path))
    if not m: return 0
    if m.group(1): return int(m.group(1)) * int(m.group(2))
    return int(m.group(3)) * int(m.group(4))
def ts_daynight(path):
    m = re.match(r"^(\d+(?:\.\d+)?)_", os.path.basename(path))
    if not m: return "?"
    try:
        h = datetime.datetime.fromtimestamp(float(m.group(1))).hour
        return "dia" if 6 <= h < 18 else "noche"
    except Exception: return "?"


@torch.no_grad()
def infer(ckpt, items, dev, bs=256):
    ck = torch.load(ckpt, map_location=dev)
    model = build_model(ck["model"], ck["num_classes"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    H, W = ck["input_size"]; preds = []
    batch = []
    def flush():
        if not batch: return
        x = torch.stack(batch).to(dev)
        idx = model(x).cpu().argmax(2).permute(1, 0)
        for r in idx: preds.append(ctc_greedy_decode(r.tolist()))
        batch.clear()
    for path, _ in items:
        im = cv2.imread(path)
        im = np.zeros((H, W, 3), np.uint8) if im is None else cv2.resize(im, (W, H))
        im = im.astype(np.float32) / 127.5 - 1.0
        batch.append(torch.from_numpy(im.transpose(2, 0, 1)))
        if len(batch) >= bs: flush()
    flush()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=120000)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--small-area", type=int, default=4300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # cargar candidatos silver
    items = []
    for split in ("train", "val", "test"):
        p = os.path.join("data", "rec_labels_%s.tsv" % split)
        if os.path.isfile(p):
            for l in open(p, encoding="utf-8").read().splitlines():
                if l:
                    path, plate = l.split("\t"); items.append((path, plate))
    random.seed(a.seed); random.shuffle(items)
    pool = items[:a.pool]
    print("pool:", len(pool), "| device:", dev)

    lp = infer("ckpt/lprnet.pt", pool, dev)
    cr = infer("ckpt/crnn.pt", pool, dev)
    print("inferencia lista")

    rows = []
    for (path, silver), pl, pc in zip(pool, lp, cr):
        s = norm_plate(silver); pl = norm_plate(pl); pc = norm_plate(pc)
        ar = area_of(path)
        if pl != pc: motivo, tier = "modelos discrepan", 0
        elif pl != s: motivo, tier = "difiere del OCR", 1
        elif ar and ar < a.small_area: motivo, tier = "placa pequeña", 2
        else: motivo, tier = "rutina", 3
        rows.append({"path": path, "fn": os.path.basename(path), "silver": s,
                     "lprnet": pl, "crnn": pc, "area": ar, "dn": ts_daynight(path),
                     "motivo": motivo, "tier": tier})

    # dedup por placa silver (una por placa), quedándose con el tier más informativo (menor)
    best = {}
    for r in rows:
        k = r["silver"] or r["fn"]
        if k not in best or r["tier"] < best[k]["tier"] or (r["tier"] == best[k]["tier"] and r["area"] < best[k]["area"]):
            best[k] = r
    uniq = list(best.values())
    # ordenar por prioridad: tier asc, luego área asc (pequeñas primero)
    uniq.sort(key=lambda r: (r["tier"], r["area"]))
    sel = uniq[:a.n]

    from collections import Counter
    print("seleccionados:", len(sel))
    print("por motivo:", dict(Counter(r["motivo"] for r in sel)))
    print("por día/noche:", dict(Counter(r["dn"] for r in sel)))
    with open("active_selection.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["fn", "path", "silver", "lprnet", "crnn", "area", "dn", "motivo"])
        w.writeheader()
        for r in sel: w.writerow({k: r[k] for k in w.fieldnames})
    print("-> active_selection.csv")


if __name__ == "__main__":
    main()
