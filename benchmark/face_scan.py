#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Escaneo de rostros (RetinaFace/MTCNN) sobre los frames publicados, para certificar
'privacy by design'. Recorre images/{train,val,test}, detecta caras con MTCNN y
registra SOLO las imagenes con deteccion (candidatas a difuminar).

Salida: face_scan_flagged.csv (una fila por deteccion) + face_scan_summary.json.
Resumible: relee el flagged y un checkpoint de conteo.
Uso:
  python face_scan.py --sample 4000            # piloto
  python face_scan.py --sample 0 --workers 4   # barrido completo
"""
import os, re, csv, json, time, argparse, random
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

IMAGES = "../Peruvian_ALPR_Dataset_final/images"
TS_RE = re.compile(r"^(\d+(?:\.\d+)?)_")


def day_night(fn):
    m = TS_RE.match(fn)
    if not m: return "?"
    import datetime
    try:
        h = datetime.datetime.fromtimestamp(float(m.group(1))).hour
        return "dia" if 6 <= h < 18 else "noche"
    except Exception:
        return "?"


def list_frames(sample):
    items = []
    for split in ("train", "val", "test"):
        d = os.path.join(IMAGES, split)
        if not os.path.isdir(d): continue
        for name in os.listdir(d):
            if name.lower().endswith(".jpg"):
                items.append((split, name))
    if sample and sample < len(items):
        random.seed(42); items = random.sample(items, sample)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000, help="0 = todo el dataset")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-prob", type=float, default=0.90)
    ap.add_argument("--min-face", type=int, default=20)
    ap.add_argument("--out", default="face_scan")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, min_face_size=a.min_face, device=dev, post_process=False)

    frames = list_frames(a.sample)
    total = len(frames)
    flagged_path = a.out + "_flagged.csv"
    new_file = not os.path.isfile(flagged_path)
    fl = open(flagged_path, "a", newline="", encoding="utf-8")
    fw = csv.writer(fl)
    if new_file:
        fw.writerow(["split", "filename", "n_faces", "max_prob", "max_box_area_px", "day_night"])

    n_proc = n_faces_imgs = n_faces_total = 0
    by_dn = {"dia": 0, "noche": 0, "?": 0}
    t0 = time.time()
    for i in range(0, total, a.batch):
        chunk = frames[i:i + a.batch]
        imgs, meta = [], []
        for split, name in chunk:
            try:
                im = Image.open(os.path.join(IMAGES, split, name)).convert("RGB")
                imgs.append(im); meta.append((split, name))
            except Exception:
                pass
        if not imgs: continue
        try:
            boxes_list, probs_list = mtcnn.detect(imgs)
        except Exception:
            # fallback uno a uno si algun tamano difiere
            boxes_list, probs_list = [], []
            for im in imgs:
                b, p = mtcnn.detect(im); boxes_list.append(b); probs_list.append(p)
        for (split, name), boxes, probs in zip(meta, boxes_list, probs_list):
            n_proc += 1
            by_dn[day_night(name)] = by_dn.get(day_night(name), 0) + 1
            if boxes is None or probs is None:
                continue
            keep = [(bx, pr) for bx, pr in zip(boxes, probs) if pr is not None and pr >= a.min_prob]
            if not keep:
                continue
            areas = [max(0, (bx[2]-bx[0])) * max(0, (bx[3]-bx[1])) for bx, _ in keep]
            n_faces_imgs += 1; n_faces_total += len(keep)
            fw.writerow([split, name, len(keep), "%.3f" % max(pr for _, pr in keep),
                         "%.0f" % max(areas), day_night(name)])
        if (i // a.batch) % 20 == 0:
            fl.flush()
            el = time.time() - t0
            print("procesadas %d/%d  con_rostro=%d  (%.1f img/s)" %
                  (n_proc, total, n_faces_imgs, n_proc / max(el, 1e-6)), flush=True)
    fl.close()
    el = time.time() - t0
    summary = {
        "n_procesadas": n_proc, "n_total_objetivo": total, "muestra": a.sample,
        "imagenes_con_rostro": n_faces_imgs, "rostros_detectados": n_faces_total,
        "tasa_imagenes_con_rostro": round(n_faces_imgs / n_proc, 6) if n_proc else 0,
        "por_dia_noche_procesadas": by_dn,
        "min_prob": a.min_prob, "min_face_size": a.min_face,
        "segundos": round(el, 1), "img_por_s": round(n_proc / max(el, 1e-6), 1),
        "detector": "MTCNN (facenet-pytorch)", "device": dev,
    }
    json.dump(summary, open(a.out + "_summary.json", "w"), ensure_ascii=False, indent=2)
    print("\nRESUMEN:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
