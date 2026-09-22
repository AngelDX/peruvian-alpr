#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renombra las claves de los checkpoints de CRNN de la ablacion al esquema de ../models.py.

models_res.CRNNRes parte la pila convolucional en cnn_a + collapse + cnn_b para
poder insertar el pooling adaptativo. models.CRNN la tiene entera en cnn. Son los
mismos pesos en el mismo orden, asi que basta con reindexar:

    cnn_a.N.*  ->  cnn.N.*      (N = 0..17, sin cambio de indice)
    cnn_b.0.*  ->  cnn.18.*     (Conv2d(512,512,2,1,0))
    cnn_b.1.*  ->  cnn.19.*     (BatchNorm2d(512))

A 32x100 los dos grafos son equivalentes: cnn_a deja la altura en 2 y el
AdaptiveMaxPool2d((2,None)) es la identidad, de modo que cnn_b ve lo mismo que
verian las capas 18 y 19 del original. El script lo comprueba numericamente antes
de guardar y aborta si no coincide.

Uso (desde benchmark/):
    python ablation_resolucion/fix_ckpt_keys.py                # convierte y verifica
    python ablation_resolucion/fix_ckpt_keys.py --dry-run      # solo informa
"""
import os
import sys
import glob
import shutil
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)

from models import build_model                 # noqa: E402
from models_res import build_model_res         # noqa: E402
from bench_utils import NUM_CLASSES            # noqa: E402


def remapear(sd):
    nuevo, tocados = {}, 0
    for k, v in sd.items():
        if k.startswith("cnn_a."):
            nuevo["cnn." + k[len("cnn_a."):]] = v
            tocados += 1
        elif k.startswith("cnn_b."):
            resto = k[len("cnn_b."):]
            idx, sufijo = resto.split(".", 1)
            nuevo["cnn.%d.%s" % (18 + int(idx), sufijo)] = v
            tocados += 1
        else:
            nuevo[k] = v
    return nuevo, tocados


def verificar(sd_viejo, sd_nuevo, input_size):
    """Comprueba que ambos grafos dan la misma salida con los mismos pesos."""
    h, w = input_size
    a = build_model_res("crnn", NUM_CLASSES, (h, w))
    a.load_state_dict(sd_viejo)
    a.eval()
    b = build_model("crnn", NUM_CLASSES)
    b.load_state_dict(sd_nuevo)
    b.eval()
    x = torch.randn(4, 3, h, w)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    if ya.shape != yb.shape:
        return False, "formas distintas %s vs %s" % (tuple(ya.shape), tuple(yb.shape))
    d = (ya - yb).abs().max().item()
    return d < 1e-4, "diferencia maxima %.2e" % d


def main():
    ap = argparse.ArgumentParser()
    # Solo 32x100: models.CRNN tiene la pila de pooling fija y a otras alturas
    # la altura colapsa antes del Conv2d de kernel 2. Las demas resoluciones de
    # la ablacion no se afinan, asi que no hace falta convertirlas.
    ap.add_argument("--patron", default=os.path.join(HERE, "resultados", "ckpt", "crnn_32x100_*.pt"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ficheros = sorted(glob.glob(a.patron))
    if not ficheros:
        raise SystemExit("no encontre checkpoints en %s" % a.patron)

    for f in ficheros:
        ck = torch.load(f, map_location="cpu")
        sd = ck["state_dict"]
        if not any(k.startswith("cnn_a.") for k in sd):
            print("%-28s ya esta en el esquema de models.py, lo dejo"
                  % os.path.basename(f), flush=True)
            continue

        h, w = tuple(ck["input_size"])
        if h % 16 != 0 or h // 16 < 2:
            print("%-28s entrada %dx%d incompatible con models.CRNN "
                  "(la altura colapsa antes del Conv2d de kernel 2), lo omito"
                  % (os.path.basename(f), h, w), flush=True)
            continue

        nuevo, tocados = remapear(sd)
        ok, detalle = verificar(sd, nuevo, (h, w))
        estado = "OK" if ok else "FALLA"
        print("%-28s %2d claves renombradas  %s  (%s)"
              % (os.path.basename(f), tocados, estado, detalle), flush=True)
        if not ok:
            raise SystemExit("la conversion no es equivalente, abortado sin tocar nada")
        if a.dry_run:
            continue

        shutil.copy2(f, f + ".bak")
        ck["state_dict"] = nuevo
        torch.save(ck, f)

    if a.dry_run:
        print("\ndry-run: no se escribio nada")
    else:
        print("\nlisto. Copias de seguridad en *.pt.bak")
        print("Siguiente:  python ablation_resolucion/finetune_all.py --modelos crnn")


if __name__ == "__main__":
    main()
