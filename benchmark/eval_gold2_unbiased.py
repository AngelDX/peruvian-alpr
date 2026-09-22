#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimacion NO SESGADA sobre gold2, corrigiendo el solape residual por reponderacion.

EL PROBLEMA. La seleccion de gold2 excluyo las placas cuyo *silver* aparecia en el
entrenamiento. Pero el silver es ruidoso: al corregirlo el anotador, 394 placas resultaron
tener una etiqueta verdadera que SI figura en train. Todas ellas (394/394) son placas
corregidas — ninguna tenia ya su silver en train.

Esto revela un limite de la particion original del corpus: agrupar por etiqueta silver **no
agrupa por vehiculo fisico**, porque el mismo coche puede recibir cadenas distintas en
pasadas distintas y acabar en particiones distintas.

NINGUN SUBCONJUNTO SIRVE TAL CUAL:
  - el total (1127) incluye 394 placas con memorizacion demostrada (+6.7 pts en el afinado);
  - el subconjunto sin solape (733) esta sesgado al revés: solo el 15.8 % son placas
    corregidas, frente al 45.3 % de la poblacion.

LA SOLUCION. Estratificar por si el anotador corrigio la placa y reponderar a la proporcion
real. Las placas marcadas OK no pueden solapar por construccion (su verdad es el silver, ya
excluido en la seleccion), de modo que ambos estratos limpios son insesgados:

    exactitud = P(OK) x exactitud(OK) + P(corregida) x exactitud(corregida sin solape)

Uso:
    python eval_gold2_unbiased.py
"""
import argparse, csv, json, math, re
from pathlib import Path

import numpy as np

from bench_utils import norm_plate
from corrector import correct as corregir
from paired_stats import predecir_ckpt, wilson

CROP_RE = re.compile(r".+_\(\s*\d+,\s*\d+,\s*3\)_(.+)\.jpg$", re.IGNORECASE)


def crudo_de_nombre(fn):
    m = CROP_RE.match(fn)
    return norm_plate(m.group(1)) if m else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/rec_labels_gold2_meta.csv")
    ap.add_argument("--train", default="data/rec_labels_train.tsv")
    ap.add_argument("--out", default="../docs/paper_materials/13_gold2_insesgado.json")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    filas = list(csv.DictReader(open(args.meta, encoding="utf-8")))
    rutas = [r["path"] for r in filas]
    verdad = [norm_plate(r["placa"]) for r in filas]
    n = len(filas)

    entren = set()
    with open(args.train, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    entren.add(p)

    solapa = np.array([v in entren for v in verdad])
    corr = np.array([r["corregida"] == "1" for r in filas])
    limpio_ok = ~corr & ~solapa            # silver correcto, sin solape posible
    limpio_corr = corr & ~solapa           # silver erroneo, sin solape

    p_corr = float(corr.mean())            # proporcion poblacional de silver erroneo
    print(f"GOLD2: {n} placas")
    print(f"  corregidas por el anotador (silver erroneo): {int(corr.sum())} = {p_corr:.4f}")
    print(f"  con solape residual: {int(solapa.sum())} ({100*solapa.mean():.1f} %), "
          f"todas ellas corregidas")
    print(f"\nEstratos insesgados:")
    print(f"  OK sin solape        : {int(limpio_ok.sum()):4d}  (peso {1-p_corr:.4f})")
    print(f"  corregida sin solape : {int(limpio_corr.sum()):4d}  (peso {p_corr:.4f})")

    # --- Sistemas
    sistemas = {}
    crudos = [crudo_de_nombre(r["fn"]) for r in filas]
    sistemas["PaddleOCR (crudo)"] = crudos
    sistemas["PaddleOCR + corrector"] = [
        norm_plate(corregir(c[:3] + "-" + c[3:])) if len(c) == 6 else "" for c in crudos]
    for nombre, ckpt in (("CRNN", "ckpt/crnn.pt"), ("LPRNet", "ckpt/lprnet.pt"),
                         ("LPRNet afinado", "ckpt/lprnet_ft_2lotes.pt")):
        if Path(ckpt).exists():
            print(f"  infiriendo {nombre} ...")
            sistemas[nombre] = predecir_ckpt(ckpt, rutas)

    rng = np.random.default_rng(42)
    i_ok = np.flatnonzero(limpio_ok)
    i_co = np.flatnonzero(limpio_corr)

    print(f"\n{'Sistema':26s} {'OK':>13s} {'corregida':>13s} {'REPONDERADA':>22s} {'ingenua':>9s}")
    resultados = {}
    for nombre, preds in sistemas.items():
        ok = np.array([norm_plate(p) == v for p, v in zip(preds, verdad)])
        a_ok = ok[limpio_ok].mean()
        a_co = ok[limpio_corr].mean() if limpio_corr.sum() else float("nan")
        est = (1 - p_corr) * a_ok + p_corr * a_co

        # IC bootstrap: se remuestrea dentro de cada estrato y tambien el peso,
        # porque p_corr tambien se estima de la muestra.
        reps = np.empty(args.boot)
        for b in range(args.boot):
            s_ok = ok[rng.choice(i_ok, size=len(i_ok))].mean()
            s_co = ok[rng.choice(i_co, size=len(i_co))].mean()
            s_p = rng.binomial(n, p_corr) / n
            reps[b] = (1 - s_p) * s_ok + s_p * s_co
        lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
        ingenua = float(ok.mean())

        print(f"{nombre:26s} {a_ok:12.4f} {a_co:13.4f}   {est:.4f} [{lo:.4f},{hi:.4f}] {ingenua:9.4f}")
        resultados[nombre] = {
            "exactitud_OK": float(a_ok), "n_OK": int(limpio_ok.sum()),
            "exactitud_corregida": float(a_co), "n_corregida": int(limpio_corr.sum()),
            "estimacion_reponderada": float(est), "ic95_bootstrap": [lo, hi],
            "estimacion_ingenua_todo_gold2": ingenua,
        }

    reporte = {
        "n_gold2": n,
        "proporcion_silver_erroneo": p_corr,
        "solape_residual": int(solapa.sum()),
        "mecanismo_del_solape": ("la seleccion excluyo por etiqueta silver; al corregirla, la "
                                 "etiqueta verdadera de 394 placas coincidio con cadenas "
                                 "presentes en train. Agrupar por silver no agrupa por vehiculo."),
        "metodo": ("estratificacion por silver correcto/erroneo y reponderacion a la "
                   "proporcion poblacional; las placas OK no pueden solapar por construccion"),
        "resultados": resultados,
    }
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {dest}")


if __name__ == "__main__":
    main()
