#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye el dataset de DETECCION para ultralytics a partir del export Roboflow.

- Convierte poligonos (class x1 y1 x2 y2 ...) al bounding box axis-aligned (class cx cy w h).
- Remapea todo a una sola clase 0 = plate (el export trae nc=2 con clase dummy '.').
- Enlaza (symlink, fallback copia) las imagenes; escribe labels convertidas y data.yaml.

Salida: benchmark/det_data/{images,labels}/{train,val,test} + det_data/data.yaml
"""
import os, shutil, glob, argparse

SRC_SPLITS = {"train": "train", "val": "valid", "test": "test"}  # dst -> src


def poly_to_bbox(vals):
    xs = vals[0::2]; ys = vals[1::2]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    w = x1 - x0; h = y1 - y0
    # clamp a [0,1]
    cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0);  h = min(max(h, 0.0), 1.0)
    return cx, cy, w, h


def convert_label(src_txt):
    out = []
    with open(src_txt, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            coords = [float(x) for x in p[1:]]
            if len(coords) % 2 != 0:
                coords = coords[:-1]  # por si hay basura
            if len(coords) == 4:  # ya es bbox cx cy w h
                cx, cy, w, h = coords
            else:  # poligono -> bbox
                cx, cy, w, h = poly_to_bbox(coords)
            if w <= 0 or h <= 0:
                continue
            out.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, w, h))
    return out


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../test2.v1i.yolov12")
    ap.add_argument("--out", default="det_data")
    a = ap.parse_args()
    stats = {}
    for dst_split, src_split in SRC_SPLITS.items():
        img_src_dir = os.path.join(a.src, src_split, "images")
        lbl_src_dir = os.path.join(a.src, src_split, "labels")
        img_dst_dir = os.path.join(a.out, "images", dst_split)
        lbl_dst_dir = os.path.join(a.out, "labels", dst_split)
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)
        n_img = n_obj = n_empty = 0
        for img in glob.glob(os.path.join(img_src_dir, "*.jpg")):
            base = os.path.splitext(os.path.basename(img))[0]
            src_txt = os.path.join(lbl_src_dir, base + ".txt")
            rows = convert_label(src_txt) if os.path.isfile(src_txt) else []
            link_or_copy(img, os.path.join(img_dst_dir, os.path.basename(img)))
            with open(os.path.join(lbl_dst_dir, base + ".txt"), "w", encoding="utf-8") as w:
                w.write("\n".join(rows) + ("\n" if rows else ""))
            n_img += 1; n_obj += len(rows); n_empty += (len(rows) == 0)
        stats[dst_split] = {"imgs": n_img, "objs": n_obj, "sin_caja": n_empty}

    yaml = os.path.join(a.out, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % os.path.abspath(a.out))
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames: ['plate']\n")
    print("STATS:", stats)
    print("data.yaml ->", os.path.abspath(yaml))


if __name__ == "__main__":
    main()
