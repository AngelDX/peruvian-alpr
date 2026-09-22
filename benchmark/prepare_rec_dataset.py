#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepara el dataset de RECONOCIMIENTO (crops -> texto de placa) para el benchmark.

Genera TSV "ruta_crop <TAB> placa6" por split:
  - train/val/test SILVER: crops del dataset organizado, etiqueta = placa corregida
    (de metadata_full con corrector), unida por clave que ignora la resolucion.
  - GOLD (test verdad-terreno): desde dataset_nuevo_validacion/labels_merged.tsv
    (placa validada 1-a-1 con SUNARP).

Uso:
  python prepare_rec_dataset.py \
     --dataset ../Peruvian_ALPR_Dataset_final \
     --meta ../dataset_final/metadata_full.csv.gz \
     --gold-dir ../dataset_nuevo_validacion \
     --out data
"""
import os, re, csv, gzip, argparse
from bench_utils import norm_plate

RES_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)|\d+x\d+")
def pair_key(fn): return RES_RE.sub("", os.path.splitext(os.path.basename(fn))[0])
def opener(p): return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def build_silver(dataset, meta, out):
    # clave -> placa corregida (de metadata)
    key2plate = {}
    with opener(meta) as fh:
        for row in csv.DictReader(fh):
            key2plate[pair_key(row["filename"])] = row.get("plate", "")
    counts = {}
    for split in ("train", "val", "test"):
        cdir = os.path.join(dataset, "crops", split)
        if not os.path.isdir(cdir):
            continue
        rows, miss, bad = [], 0, 0
        for f in os.listdir(cdir):
            if not f.lower().endswith(".jpg"):
                continue
            plate = key2plate.get(pair_key(f))
            if plate is None:
                miss += 1; continue
            p6 = norm_plate(plate)
            if not p6:
                bad += 1; continue
            rows.append((os.path.abspath(os.path.join(cdir, f)), p6))
        with open(os.path.join(out, "rec_labels_%s.tsv" % split), "w", encoding="utf-8") as w:
            for path, p in rows:
                w.write("%s\t%s\n" % (path, p))
        counts[split] = {"n": len(rows), "sin_meta": miss, "placa_invalida": bad}
    return counts


def build_gold(gold_dir, out):
    tsv = os.path.join(gold_dir, "labels_merged.tsv")
    imgs = os.path.join(gold_dir, "images_plates")
    rows, bad = [], 0
    with open(tsv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = row["nombre_archivo"].strip().strip('"')
            p6 = norm_plate(row["placa_real"])
            path = os.path.join(imgs, fn)
            if not p6 or not os.path.isfile(path):
                bad += 1; continue
            rows.append((os.path.abspath(path), p6))
    with open(os.path.join(out, "rec_labels_gold.tsv"), "w", encoding="utf-8") as w:
        for path, p in rows:
            w.write("%s\t%s\n" % (path, p))
    return {"n": len(rows), "descartadas": bad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../Peruvian_ALPR_Dataset_final")
    ap.add_argument("--meta", default="../dataset_final/metadata_full.csv.gz")
    ap.add_argument("--gold-dir", default="../dataset_nuevo_validacion")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    silver = build_silver(a.dataset, a.meta, a.out)
    gold = build_gold(a.gold_dir, a.out)
    print("SILVER:", silver)
    print("GOLD:", gold)
    print("TSV escritos en", os.path.abspath(a.out))
