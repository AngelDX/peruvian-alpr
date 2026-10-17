#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de tipos de matricula peruana en condiciones reales de captura.

Muestra recortes autenticos del conjunto, agrupados por la clase de placa que asignaron los
anotadores (color y franja segun la normativa nacional) y alternando dia y night dentro de cada
fila. El objetivo es que el lector vea la variedad real a la que se enfrenta el reconocedor: no
solo el formato de caracteres, sino el estado de la placa, el marco, la suciedad, el angulo y
la iluminacion.

Los recortes se muestran a su tamano relativo real dentro de cada celda, de modo que tambien se
aprecie la disparidad de resolucion entre capturas.

Uso:
    python fig_tipos_placa.py
"""
import os
import argparse, csv, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from redaccion import tapar_caracter, texto_redactado

# Rutas resueltas desde la ubicacion del script, no desde el directorio de trabajo.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_AQUI)
_TEX = os.path.abspath(os.path.join(_BENCH, "..", "docs", "paper_sensors", "tex"))


def _b(*p):
    return os.path.join(_BENCH, *p)


def _t(nombre):
    return os.path.join(_TEX, nombre)



# Denominacion segun la clase anotada, con su significado en la normativa peruana.
NOMBRE_EN = {"BLANCA": "White", "AMARILLA": "Yellow", "FRANJA_VERDE": "Green band",
             "FRANJA_AMARILLA": "Yellow band", "FRANJA_NARANJA": "Orange band"}
TIPOS = [
    ("BLANCA", "Private"),
    ("AMARILLA", "Public service / freight"),
    ("FRANJA_VERDE", "Public transport"),
    ("FRANJA_AMARILLA", "Taxi / shared ride"),
    ("FRANJA_NARANJA", "Special service"),
]


def _remap(p):
    """Permite ejecutar desde Linux sobre las rutas Windows guardadas en el CSV."""
    if os.path.exists(p):
        return p
    q = p.replace(chr(92), "/")
    pref, dest = "F:/DATAPLACAS/", "/sessions/pensive-great-pascal/mnt/DATAPLACAS/"
    if q.startswith(pref):
        return dest + q[len(pref):]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=_b("data", "rec_labels_gold2_meta.csv"))
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=_t("fig1_plate_types_en.png"))
    a = ap.parse_args()

    filas = list(csv.DictReader(open(a.meta, encoding="utf-8")))
    por_tipo = defaultdict(lambda: defaultdict(list))
    for r in filas:
        r["path"] = _remap(r["path"])
        if Path(r["path"]).exists():
            por_tipo[r["tipo_placa"]][r["luz"]].append(r)

    rng = random.Random(a.seed)
    nf, nc = len(TIPOS), a.cols
    fig, axes = plt.subplots(nf, nc, figsize=(nc * 2.05, nf * 1.16), dpi=300)

    for i, (tipo, glosa) in enumerate(TIPOS):
        disponibles = por_tipo.get(tipo, {})
        # Se alterna dia y night para que cada fila muestre ambas condiciones.
        secuencia = []
        dia = disponibles.get("dia", [])[:]
        night = disponibles.get("noche", [])[:] + disponibles.get("crepusculo", [])[:]
        rng.shuffle(dia)
        rng.shuffle(night)
        objetivo_night = max(1, nc // 3)
        secuencia = dia[: nc - objetivo_night] + night[:objetivo_night]
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
            etq = "night" if r["luz"] == "noche" else ("twil." if r["luz"] == "crepusculo" else "day")
            ax.set_xlabel(f"{texto_redactado(r['placa'])}  ·  {etq}",
                          fontsize=5.6, labelpad=1.5, color="#3d444d")
            if j == 0:
                ax.set_ylabel(f"{NOMBRE_EN.get(tipo, tipo.replace('_',' ').title())}\n{glosa}",
                              fontsize=6.4, rotation=0, ha="right", va="center",
                              labelpad=52, color="#161b21")

    fig.suptitle("Peruvian license plate types under real capture conditions",
                 fontsize=10.5, y=0.995, x=0.53, weight="bold")
    fig.text(0.53, -0.012,
             "Authentic crops from the dataset, at true relative scale. Each row corresponds to one "
             "plate class; daytime and nighttime captures alternate. For privacy, the first character "
             "after the hyphen is masked.",
             ha="center", fontsize=6.4, color="#5a6572")
    fig.subplots_adjust(left=0.15, right=0.995, top=0.955, bottom=0.035, wspace=0.06, hspace=0.42)

    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    print(f"Figure: {dest}")
    for tipo, _ in TIPOS:
        d = por_tipo.get(tipo, {})
        print(f"  {tipo:18s} dia={len(d.get('dia', [])):4d} "
              f"night={len(d.get('noche', [])):3d} crep={len(d.get('crepusculo', [])):2d}")


if __name__ == "__main__":
    main()
