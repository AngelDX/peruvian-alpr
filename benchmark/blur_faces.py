#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera el espejo anonimizado COMPLETO del set de frames para publicacion.

- Los frames marcados por face_scan.py se difuminan (desenfoque gaussiano + margen).
- El resto se enlaza por hardlink (o copia si el FS no lo soporta) SIN duplicar disco.
Resultado: images_anonymized/{train,val,test} = reemplazo drop-in de images/.

Uso:
  python blur_faces.py --flagged face_scan_flagged.csv --min-prob 0.90
  python blur_faces.py --only-flagged      # solo difumina las marcadas (sin espejo completo)
"""
import os, csv, cv2, argparse, numpy as np, torch
from facenet_pytorch import MTCNN
from PIL import Image

IMAGES = "../Peruvian_ALPR_Dataset_final/images"
OUT = "../Peruvian_ALPR_Dataset_final/images_anonymized"
SPLITS = ("train", "val", "test")


def blur_region(img, x0, y0, x1, y1, pad=0.25):
    h, w = img.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - pad * bw)); y0 = max(0, int(y0 - pad * bh))
    x1 = min(w, int(x1 + pad * bw)); y1 = min(h, int(y1 + pad * bh))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    k = max(15, (max(roi.shape[:2]) // 2) | 1)
    img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)               # hardlink: 0 bytes extra
    except OSError:
        import shutil; shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", default="face_scan_flagged.csv")
    ap.add_argument("--min-prob", type=float, default=0.90)
    ap.add_argument("--min-face", type=int, default=20)
    ap.add_argument("--only-flagged", action="store_true", help="no crear espejo completo")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, min_face_size=a.min_face, device=dev, post_process=False)

    flagged = {s: set() for s in SPLITS}
    for r in csv.DictReader(open(a.flagged, encoding="utf-8")):
        flagged.setdefault(r["split"], set()).add(r["filename"])
    n_flag = sum(len(v) for v in flagged.values())

    man = open("blur_manifest.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(man); mw.writerow(["split", "filename", "n_cajas_difuminadas"])
    blurred = boxes = linked = 0
    for split in SPLITS:
        srcdir = os.path.join(IMAGES, split)
        if not os.path.isdir(srcdir):
            continue
        outdir = os.path.join(OUT, split); os.makedirs(outdir, exist_ok=True)
        names = os.listdir(srcdir) if not a.only_flagged else list(flagged.get(split, ()))
        for name in names:
            src = os.path.join(srcdir, name); dst = os.path.join(outdir, name)
            if name in flagged.get(split, ()):
                pil = Image.open(src).convert("RGB")
                bx, pr = mtcnn.detect(pil)
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                n = 0
                if bx is not None:
                    for b, p in zip(bx, pr):
                        if p is not None and p >= a.min_prob:
                            blur_region(img, *b); n += 1
                cv2.imwrite(dst, img); mw.writerow([split, name, n]); blurred += 1; boxes += n
                if blurred % 200 == 0:
                    man.flush(); print("difuminadas %d/%d  cajas=%d  enlazadas=%d" %
                                       (blurred, n_flag, boxes, linked), flush=True)
            elif not a.only_flagged:
                link_or_copy(src, dst); linked += 1
    man.close()
    print("\nLISTO: %d frames difuminados (%d rostros), %d enlazados sin cambio -> %s" %
          (blurred, boxes, linked, OUT))
    print("El espejo %s es reemplazo drop-in de images/ para la publicacion." % OUT)


if __name__ == "__main__":
    main()
