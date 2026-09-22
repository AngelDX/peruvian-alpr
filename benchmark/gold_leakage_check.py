#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mide si el solape de placas entre el gold y el train infla las metricas de reconocimiento.

EL PROBLEMA. La particion del corpus de 579 292 frames se hizo agrupando por placa, de modo
que ninguna placa cruza train/val/test. Pero el *gold* SUNARP procede de un lote de
verificacion distinto (`dataset_nuevo_validacion`) y sus placas NO se excluyeron del train
del reconocedor: 424 de 524 placas del gold (80.9 %) aparecen tambien en entrenamiento, con
otros frames del mismo vehiculo. El modelo pudo memorizar esas matriculas concretas.

LA PRUEBA. Se parte el gold en dos subconjuntos disjuntos -- placas VISTAS en train y placas
NO VISTAS -- y se compara la exactitud en cada uno. Son grupos **no apareados** (placas
distintas, tamanos distintos), asi que la diferencia se acota con el IC de Newcombe para dos
proporciones independientes, no con McNemar.

Interpretacion: si la exactitud sobre las placas no vistas se mantiene, el solape no explica
el rendimiento y las cifras del articulo son defendibles (declarando el solape). Si cae
mucho, las metricas globales estan infladas y hay que reportar el subconjunto limpio.

Uso:
    python gold_leakage_check.py --out ../docs/paper_materials/10_solape_gold_train.json
"""
import argparse, csv, json, math, os
from pathlib import Path

import numpy as np

from bench_utils import norm_plate
from corrector import correct as corregir
from paired_stats import (cargar_gold, cargar_paddle, predecir_ckpt, wilson)


def placas_de(tsv):
    s = set()
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    s.add(p)
    return s


def newcombe(k1, n1, k2, n2):
    """IC 95 % de la diferencia de dos proporciones independientes."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/rec_labels_gold.tsv")
    ap.add_argument("--train", default="data/rec_labels_train.tsv")
    ap.add_argument("--paddle", default="../dataset_nuevo_validacion/resultados_verificacion.tsv")
    ap.add_argument("--out", default="../docs/paper_materials/10_solape_gold_train.json")
    args = ap.parse_args()

    filas = cargar_gold(args.gold)
    rutas = [r for r, _ in filas]
    verdad = [v for _, v in filas]
    vistas = placas_de(args.train)

    en_train = np.array([v in vistas for v in verdad])
    n_v, n_nv = int(en_train.sum()), int((~en_train).sum())
    placas_gold = set(verdad)
    print(f"Gold: {len(filas)} recortes, {len(placas_gold)} placas unicas")
    print(f"  placas del gold presentes en train: {len(placas_gold & vistas)}/{len(placas_gold)}"
          f" ({100*len(placas_gold & vistas)/len(placas_gold):.1f} %)")
    print(f"  recortes con placa VISTA en train : {n_v}")
    print(f"  recortes con placa NO VISTA       : {n_nv}\n")

    crudas, corregidas, verdad_paddle, faltan = cargar_paddle(args.paddle, rutas)
    if faltan:
        raise SystemExit(f"{faltan} placas del gold no aparecen en {args.paddle}")

    sistemas = {"PaddleOCR + corrector": corregidas}
    for nombre, ckpt in (("CRNN", "ckpt/crnn.pt"),
                         ("LPRNet", "ckpt/lprnet.pt"),
                         ("LPRNet afinado", "ckpt/lprnet_ft_2lotes.pt")):
        if Path(ckpt).exists():
            print(f"  infiriendo {nombre} ...")
            sistemas[nombre] = predecir_ckpt(ckpt, rutas)

    print(f"\n{'Sistema':26s} {'vista en train':>22s} {'NO vista':>22s} {'diferencia':>26s}")
    filas_out = []
    for nombre, preds in sistemas.items():
        ok = np.array([p == v for p, v in zip(preds, verdad)])
        k1, k2 = int(ok[en_train].sum()), int(ok[~en_train].sum())
        a1, a2 = k1 / n_v, k2 / n_nv
        l1, u1 = wilson(k1, n_v)
        l2, u2 = wilson(k2, n_nv)
        lo, hi = newcombe(k1, n_v, k2, n_nv)
        print(f"{nombre:26s} {k1:4d}/{n_v:<4d} {a1:.4f}      {k2:3d}/{n_nv:<3d} {a2:.4f}"
              f"      {a1-a2:+.4f} [{lo:+.4f},{hi:+.4f}]")
        filas_out.append({
            "sistema": nombre,
            "placa_vista_en_train": {"aciertos": k1, "n": n_v, "exactitud": a1,
                                     "ic95_wilson": [l1, u1]},
            "placa_no_vista": {"aciertos": k2, "n": n_nv, "exactitud": a2,
                               "ic95_wilson": [l2, u2]},
            "diferencia": a1 - a2,
            "ic95_newcombe": [lo, hi],
            "diferencia_significativa_0.05": not (lo <= 0 <= hi),
        })

    reporte = {
        "n_recortes_gold": len(filas),
        "n_placas_gold": len(placas_gold),
        "placas_gold_en_train": len(placas_gold & vistas),
        "proporcion_placas_solapadas": len(placas_gold & vistas) / len(placas_gold),
        "n_recortes_placa_vista": n_v,
        "n_recortes_placa_no_vista": n_nv,
        "diseno": ("subgrupos NO apareados del gold (placas distintas); "
                   "IC de la diferencia por Newcombe para proporciones independientes"),
        "resultados": filas_out,
    }
    destino = Path(args.out).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {destino}")


if __name__ == "__main__":
    main()
