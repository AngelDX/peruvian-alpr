#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de tipos de matricula peruana en condiciones reales de captura.

Muestra recortes autenticos del conjunto, agrupados por la clase de placa que asignaron los
anotadores (color y franja segun la normativa nacional) y alternando dia y noche dentro de cada
fila. El objetivo es que el lector vea la variedad real a la que se enfrenta el reconocedor: no
solo el formato de caracteres, sino el estado de la placa, el marco, la suciedad, el angulo y
la iluminacion.

Los recortes se muestran a su tamano relativo real dentro de cada celda, de modo que tambien se
aprecie la disparidad de resolucion entre capturas.

Uso:
    python fig_tipos_placa.py
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from redaccion import tapar_caracter, texto_redactado

# Denominacion segun la clase anotada, con su significado en la normativa peruana.
TIPOS = [
    ("BLANCA", "Particular"),
    ("AMARILLA", "Servicio público / carga"),
    ("FRANJA_VERDE", "Transporte público"),
    ("FRANJA_AMARILLA", "Taxi / colectivo"),
    ("FRANJA_NARANJA", "Servicio especial"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/rec_labels_gold2_meta.csv")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="../docs/figures/fig1_tipos_placa_es.png")
    a = ap.parse_args()

    filas = list(csv.DictReader(open(a.meta, encoding="utf-8")))
    por_tipo = defaultdict(lambda: defaultdict(list))
    for r in filas:
        if Path(r["path"]).exists():
            por_tipo[r["tipo_placa"]][r["luz"]].append(r)

    rng = random.Random(a.seed)
    nf, nc = len(TIPOS), a.cols
    fig, axes = plt.subplots(nf, nc, figsize=(nc * 2.05, nf * 1.16), dpi=300)

    for i, (tipo, glosa) in enumerate(TIPOS):
        disponibles = por_tipo.get(tipo, {})
        # Se alterna dia y noche para que cada fila muestre ambas condiciones.
        secuencia = []
        dia = disponibles.get("dia", [])[:]
        noche = disponibles.get("noche", [])[:] + disponibles.get("crepusculo", [])[:]
        rng.shuffle(dia)
        rng.shuffle(noche)
        objetivo_noche = max(1, nc // 3)
        secuencia = dia[: nc - objetivo_noche] + noche[:objetivo_noche]
        secuencia = secuencia[:nc]
        rng.shuffle(secuencia)

        for j in range(nc):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c9d1d9"); s.set_linewidth(0.6)
            if j >= len(secuencia):
                ax.set_facecolor("#f2f4f7")
                continue
            r = secuencia[j]
            try:
                # Redacción parcial: se cubre el primer carácter tras el guion para que la
                # matrícula no sea consultable en el registro oficial.
                im = tapar_caracter(Image.open(r["path"]).convert("RGB"))
            except Exception:
                ax.set_facecolor("#f2f4f7"); continue
            ax.imshow(im)
            etq = "noche" if r["luz"] == "noche" else ("crep." if r["luz"] == "crepusculo" else "día")
            ax.set_xlabel(f"{texto_redactado(r['placa'])}  ·  {etq}",
                          fontsize=5.6, labelpad=1.5, color="#3d444d")
            if j == 0:
                ax.set_ylabel(f"{tipo.replace('_', ' ').title()}\n{glosa}",
                              fontsize=6.4, rotation=0, ha="right", va="center",
                              labelpad=52, color="#161b21")

    fig.suptitle("Tipos de matrícula peruana en condiciones reales de captura",
                 fontsize=10.5, y=0.995, x=0.53, weight="bold")
    fig.text(0.53, -0.012,
             "Recortes auténticos del conjunto, a escala relativa real. Cada fila corresponde a una "
             "clase de placa; se alternan capturas diurnas y nocturnas. Por privacidad se cubre el "
             "primer carácter posterior al guion.",
             ha="center", fontsize=6.4, color="#5a6572")
    fig.subplots_adjust(left=0.15, right=0.995, top=0.955, bottom=0.035, wspace=0.06, hspace=0.42)

    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    print(f"Figura: {dest}")
    for tipo, _ in TIPOS:
        d = por_tipo.get(tipo, {})
        print(f"  {tipo:18s} dia={len(d.get('dia', [])):4d} "
              f"noche={len(d.get('noche', [])):3d} crep={len(d.get('crepusculo', [])):2d}")


if __name__ == "__main__":
    main()
