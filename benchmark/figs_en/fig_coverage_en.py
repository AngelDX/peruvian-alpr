#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New figure for revision 1: seasonal and diurnal coverage of the corpus.

Reviewer 1 asked for a fuller account of the environmental conditions, weather
and lighting in particular. The site has a marked wet season, so what matters is
not a nominal date range but how the captures distribute across months and hours.
This figure shows both, with the wet season and the solar-elevation bands marked.

Input: metadata_stats.json, produced by the dataset audit.

Usage (from benchmark/):
    python figs_en/fig_coverage_en.py
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, GRIS, TINTA = "#26527d", "#5b6b7c", "#16202b"
LLUVIA, SECA = "#cfe0f5", "#f6efe0"


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.dirname(aqui)
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(bench, "metadata_stats.json"))
    ap.add_argument("--out", default=os.path.join(bench, "..", "docs", "paper_sensors",
                                                  "tex", "fig_coverage_en"))
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as f:
        d = json.load(f)
    meses = dict(sorted(d["dist"]["month"].items()))
    fuente = d.get("hours") or d["dist"]["hours"]
    horas = {int(k): v for k, v in fuente.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.0), dpi=300,
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # ---------------------------------------------------------------- (a) por mes
    etiquetas = list(meses)
    valores = [meses[m] / 1000.0 for m in etiquetas]
    # Temporada de lluvias en el altiplano: diciembre a marzo.
    humedo = [m.split("-")[1] in ("12", "01", "02", "03") for m in etiquetas]
    x = np.arange(len(etiquetas))
    ax1.bar(x, valores, color=[LLUVIA if h else SECA for h in humedo],
            edgecolor=AZUL, linewidth=0.8)
    for xi, v, h in zip(x, valores, humedo):
        ax1.annotate("%.0f" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=6.4, color=GRIS)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[2:] for m in etiquetas], fontsize=7.2, rotation=90)
    ax1.set_ylabel("captures (thousands)")
    ax1.set_xlabel("month of operation")
    ax1.set_title("(a) Seasonal coverage", fontsize=10, loc="left")
    ax1.grid(axis="y", alpha=.3)
    ax1.set_axisbelow(True)
    izq = plt.Rectangle((0, 0), 1, 1, fc=LLUVIA, ec=AZUL, lw=.8)
    der = plt.Rectangle((0, 0), 1, 1, fc=SECA, ec=AZUL, lw=.8)
    ax1.legend([izq, der], ["wet season (Dec–Mar)", "dry season (Apr–Nov)"],
               fontsize=7.6, frameon=False, loc="upper right")

    # ---------------------------------------------------------------- (b) por hora
    h = np.arange(24)
    v = np.array([horas.get(i, 0) / 1000.0 for i in h])
    ax2.bar(h, v, color="#dce7f6", edgecolor=AZUL, linewidth=0.8)
    # Franjas lumínicas aproximadas para la latitud y longitud del emplazamiento.
    ax2.axvspan(-0.5, 5.5, color="#2b3a4a", alpha=.10, lw=0)
    ax2.axvspan(17.5, 23.5, color="#2b3a4a", alpha=.10, lw=0)
    ax2.annotate("night", (2.5, v.max() * .88), ha="center", fontsize=8, color=GRIS)
    ax2.annotate("night", (20.5, v.max() * .88), ha="center", fontsize=8, color=GRIS)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.6, 23.6)
    ax2.set_xlabel("hour of day (local time)")
    ax2.set_ylabel("captures (thousands)")
    ax2.set_title("(b) Diurnal coverage", fontsize=10, loc="left")
    ax2.grid(axis="y", alpha=.3)
    ax2.set_axisbelow(True)

    fig.text(.006, .022,
             "Counts are taken over the 579,292 retained plate crops. Shading in (a) marks the "
             "altiplano wet season; the drop from December to February combines rainfall with the\n"
             "academic recess. Shading in (b) marks the hours that lie below the horizon all year "
             "at this latitude.",
             fontsize=7.2, color=GRIS, linespacing=1.4)
    fig.subplots_adjust(left=.055, right=.995, top=.92, bottom=.28, wspace=.22)

    destino = os.path.abspath(a.out)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (destino, ext), dpi=300, facecolor="white")
    print("figure ->", destino + ".png / .pdf")


if __name__ == "__main__":
    main()
