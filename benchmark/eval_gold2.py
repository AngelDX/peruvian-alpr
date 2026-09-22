#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluacion de todos los reconocedores sobre GOLD2, el conjunto limpio.

Gold2 corrige los tres sesgos que inflaban las metricas del gold original:
  1. cero solape de placas con el entrenamiento (se verifica aqui, no se asume);
  2. distribucion de frecuencias igual a la del corpus (el gold original tenia el 74 % de
     sus placas en vehiculos de 50+ pasadas, frente al 8 % de la poblacion real);
  3. ningun checkpoint fue seleccionado midiendo sobre el.

Por tanto las cifras que salen de aqui son las primeras estimaciones no sesgadas del
proyecto, y son las que deben ir al articulo como resultado principal.

Sistemas evaluados. El OCR generico se reconstruye sin volver a ejecutarlo: la lectura cruda
de PaddleOCR esta en el nombre de cada recorte, y aplicarle el corrector determinista
reproduce la etiqueta silver en 1126/1127 casos (verificado).

Uso:
    python eval_gold2.py
"""
import argparse, csv, json, math, os, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench_utils import norm_plate
from corrector import correct as corregir
from paired_stats import (predecir_ckpt, mcnemar_exacto, bootstrap_diferencia, wilson)

CROP_RE = re.compile(r".+_\(\s*\d+,\s*\d+,\s*3\)_(.+)\.jpg$", re.IGNORECASE)


def newcombe(k1, n1, k2, n2):
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def crudo_de_nombre(fn):
    m = CROP_RE.match(fn)
    return norm_plate(m.group(1)) if m else ""


def bucket_frec(n):
    n = int(n or 0)
    return "1-2" if n <= 2 else "3-9" if n < 10 else "10-49" if n < 50 else "50+"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/rec_labels_gold2_meta.csv")
    ap.add_argument("--train", default="data/rec_labels_train.tsv")
    ap.add_argument("--out", default="../docs/paper_materials/12_resultados_gold2.json")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    filas = list(csv.DictReader(open(args.meta, encoding="utf-8")))
    rutas = [r["path"] for r in filas]
    verdad = [norm_plate(r["placa"]) for r in filas]
    n = len(filas)
    print(f"GOLD2: {n} placas verificadas contra SUNARP\n")

    # --- Verificacion de limpieza: no se asume, se comprueba.
    entren = set()
    with open(args.train, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    entren.add(p)
    solape = sum(1 for v in verdad if v in entren)
    print(f"Verificacion de solape con el entrenamiento: {solape}/{n} placas "
          f"({100*solape/n:.2f} %)")
    if solape:
        print("  AVISO: hay solape; las cifras siguientes no son del todo limpias")
    else:
        print("  Sin solape. Las metricas son estimaciones no sesgadas.")

    perfil = Counter(bucket_frec(r["frames_de_la_placa"]) for r in filas)
    print("Frecuencia de la placa en el corpus: " +
          "  ".join(f"{k}={100*perfil[k]/n:.1f}%" for k in ("1-2", "3-9", "10-49", "50+")))
    print("  (corpus real: 1-2=52.8%  3-9=24.1%  10-49=15.1%  50+=7.9%)\n")

    # --- Sistemas
    sistemas = {}
    crudos = [crudo_de_nombre(r["fn"]) for r in filas]
    sistemas["PaddleOCR (crudo)"] = crudos
    sistemas["PaddleOCR + corrector"] = [
        norm_plate(corregir(c[:3] + "-" + c[3:])) if len(c) == 6 else "" for c in crudos
    ]
    for nombre, ckpt in (("CRNN", "ckpt/crnn.pt"),
                         ("LPRNet", "ckpt/lprnet.pt"),
                         ("LPRNet afinado", "ckpt/lprnet_ft_2lotes.pt")):
        if Path(ckpt).exists():
            print(f"  infiriendo {nombre} ...")
            sistemas[nombre] = predecir_ckpt(ckpt, rutas)
    if "LPRNet afinado" in sistemas:
        sistemas["LPRNet afinado + corrector"] = [
            norm_plate(corregir(p[:3] + "-" + p[3:])) if len(p) == 6 else p
            for p in sistemas["LPRNet afinado"]
        ]

    aciertos = {k: np.array([norm_plate(p) == v for p, v in zip(v_, verdad)])
                for k, v_ in sistemas.items()}

    # --- Resultado principal
    print(f"\n{'Sistema':30s} {'Aciertos':>11s} {'Exactitud':>10s} {'IC 95 % Wilson':>21s}")
    resumen = {}
    for nombre, ok in aciertos.items():
        k = int(ok.sum())
        lo, hi = wilson(k, n)
        resumen[nombre] = {"aciertos": k, "n": n, "exactitud": k / n, "ic95": [lo, hi]}
        print(f"{nombre:30s} {k:5d}/{n:<5d} {k/n:9.4f}   [{lo:.4f}, {hi:.4f}]")

    # --- Comparaciones apareadas
    pares = [
        ("LPRNet afinado", "PaddleOCR + corrector", "El estudiante frente al maestro"),
        ("LPRNet afinado", "LPRNet", "Efecto del ajuste fino"),
        ("PaddleOCR + corrector", "PaddleOCR (crudo)", "Aporte del corrector"),
        ("LPRNet", "CRNN", "Arquitecturas compactas"),
        ("LPRNet afinado + corrector", "LPRNet afinado", "Corrector tras el ajuste fino"),
    ]
    print(f"\n{'Comparacion (A vs B)':50s} {'dif':>8s} {'IC 95 %':>20s} {'b10/b01':>9s} {'p':>10s}")
    comparaciones = []
    for a_n, b_n, hip in pares:
        if a_n not in aciertos or b_n not in aciertos:
            continue
        a, b = aciertos[a_n], aciertos[b_n]
        mc = mcnemar_exacto(a, b)
        lo, hi = bootstrap_diferencia(a, b, n_iter=args.boot)
        dif = float(a.mean() - b.mean())
        print(f"{a_n + ' vs ' + b_n:50s} {dif:+8.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{mc['b10']:4d}/{mc['b01']:<4d} {mc['p']:10.2e}")
        comparaciones.append({"A": a_n, "B": b_n, "hipotesis": hip, "diferencia": dif,
                              "ic95_bootstrap": [lo, hi], "mcnemar": mc,
                              "significativo_0.05": mc["p"] < 0.05})

    # --- Estratificaciones (subgrupos NO apareados)
    estratos = {
        "clase luminica": [(c, np.array([r["luz"] == c for r in filas]))
                           for c in ("dia", "crepusculo", "noche")],
        "frecuencia de la placa": [(c, np.array([bucket_frec(r["frames_de_la_placa"]) == c
                                                 for r in filas]))
                                   for c in ("1-2", "3-9", "10-49", "50+")],
    }
    estrat_out = {}
    for titulo, grupos in estratos.items():
        print(f"\nExactitud por {titulo}:")
        cab = "  ".join(f"{c:>14s}" for c, _ in grupos)
        print(f"  {'Sistema':28s} {cab}")
        estrat_out[titulo] = {}
        for nombre, ok in aciertos.items():
            celdas, datos = [], {}
            for c, m in grupos:
                if m.sum() == 0:
                    celdas.append(f"{'—':>14s}"); continue
                k = int(ok[m].sum()); nn = int(m.sum())
                celdas.append(f"{k}/{nn} {k/nn:.3f}".rjust(14))
                datos[c] = {"aciertos": k, "n": nn, "exactitud": k / nn}
            print(f"  {nombre:28s} " + "  ".join(celdas))
            estrat_out[titulo][nombre] = datos

    # --- Comparacion con el gold original
    print("\nContraste con el gold original (n = 782, con los tres sesgos):")
    previo = {"PaddleOCR (crudo)": 0.6509, "PaddleOCR + corrector": 0.9540, "CRNN": 0.9118,
              "LPRNet": 0.9271, "LPRNet afinado": 0.9744, "LPRNet afinado + corrector": 0.9757}
    print(f"  {'Sistema':30s} {'gold (782)':>11s} {'gold2':>9s} {'cambio':>9s}")
    for nombre in sistemas:
        if nombre in previo:
            ant, act = previo[nombre], resumen[nombre]["exactitud"]
            print(f"  {nombre:30s} {ant:10.4f} {act:9.4f} {act-ant:+9.4f}")

    reporte = {
        "n": n, "solape_con_train": solape,
        "diseno": "gold limpio: sin solape con entrenamiento, frecuencias como el corpus",
        "perfil_frecuencia": dict(perfil),
        "sistemas": resumen,
        "comparaciones_apareadas": comparaciones,
        "estratificaciones": estrat_out,
        "gold_original_referencia": previo,
    }
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {dest}")


if __name__ == "__main__":
    main()
