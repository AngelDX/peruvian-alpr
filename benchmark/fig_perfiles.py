#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura 5: exactitud por sistema y perfiles por subgrupo.

Sustituye a la antigua fig5_inferencial, que repetia el bosque de diferencias
que ahora cubre la Figura 6 de caja y bigotes. Aqui se muestran las dos
interacciones que el modelo factorial declara significativas, que es lo que la
tabla no deja ver de un vistazo: como responde cada sistema a la frecuencia de
paso y a la condicion luminica.

Entrada: results/cifras_32x100.json, que genera regenerar_cifras_32x100.py.

Uso (desde benchmark/):
    python fig_perfiles.py
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDEN = ["LPRNet afinado", "CRNN afinado", "LPRNet", "CRNN",
         "PaddleOCR + corrector", "PaddleOCR crudo"]
CORTO = {"LPRNet afinado": "LPRNet afinado", "CRNN afinado": "CRNN afinado",
         "LPRNet": "LPRNet (silver)", "CRNN": "CRNN (silver)",
         "PaddleOCR + corrector": "PaddleOCR + corrector",
         "PaddleOCR crudo": "PaddleOCR crudo"}
COLOR = {"LPRNet afinado": "#1a7f4b", "CRNN afinado": "#57b894",
         "LPRNet": "#26527d", "CRNN": "#7ba3cc",
         "PaddleOCR + corrector": "#b5670a", "PaddleOCR crudo": "#c81e2c"}
MARCA = {"LPRNet afinado": "o", "CRNN afinado": "s", "LPRNet": "^",
         "CRNN": "v", "PaddleOCR + corrector": "D", "PaddleOCR crudo": "X"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/cifras_32x100.json")
    ap.add_argument("--out", default="../docs/paper_sensors/tex/fig5_perfiles_es")
    a = ap.parse_args()

    if not os.path.exists(a.json):
        raise SystemExit("falta %s. Genéralo con regenerar_cifras_32x100.py" % a.json)
    with open(a.json, encoding="utf-8") as f:
        d = json.load(f)

    gb = d["gold_B"]
    dg = d.get("desglose", {})
    sistemas = [s for s in ORDEN if s in gb]

    fig = plt.figure(figsize=(13.8, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=.3,
                          left=.115, right=.995, top=.88, bottom=.28)

    # ---------------- (a) exactitud con IC de Wilson ----------------
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(sistemas))[::-1]
    for yi, s in zip(y, sistemas):
        v = gb[s]
        lo, hi = v["ic95"]
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=COLOR[s], lw=2.2,
                solid_capstyle="round", zorder=2)
        ax.plot(100 * v["exacta"], yi, MARCA[s], color=COLOR[s], ms=7, zorder=3)
        if v["corridas"] > 1:
            ax.errorbar(100 * v["exacta"], yi, xerr=100 * v["sd"], fmt="none",
                        ecolor=COLOR[s], elinewidth=3.2, alpha=.35, capsize=0, zorder=1)
        ax.annotate("%.1f" % (100 * v["exacta"]), (100 * v["exacta"], yi),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=COLOR[s], fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([CORTO[s] for s in sistemas], fontsize=8.5)
    ax.set_xlabel("coincidencia exacta en Gold-B (%)")
    ax.set_title("(a) Exactitud, IC 95 % de Wilson", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=.3)
    ax.set_ylim(-.7, len(sistemas) - .3)

    # ---------------- (b) y (c) perfiles por subgrupo ----------------
    paneles = [("frecuencia", "(b) Por número de pasadas del vehículo", "pasadas registradas"),
               ("luz", "(c) Por condición lumínica", "clase lumínica")]
    for k, (clave, titulo, xlab) in enumerate(paneles, start=1):
        ax = fig.add_subplot(gs[k])
        if clave not in dg:
            ax.text(.5, .5, "sin datos de %s" % clave, ha="center", va="center",
                    transform=ax.transAxes, color="#8a8a8a")
            ax.set_axis_off()
            continue
        niveles = list(dg[clave]["n"])
        x = np.arange(len(niveles))
        for s in sistemas:
            if s not in dg[clave]["sistemas"]:
                continue
            vals = [100 * dg[clave]["sistemas"][s][lv] for lv in niveles]
            ax.plot(x, vals, marker=MARCA[s], color=COLOR[s], lw=1.8, ms=5.5,
                    label=CORTO[s])
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n(n=%d)" % (lv, dg[clave]["n"][lv]) for lv in niveles],
                           fontsize=8)
        ax.set_xlabel(xlab)
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.grid(alpha=.3)
        ax.set_ylim(30, 102)
        if k == 1:
            ax.set_ylabel("coincidencia exacta (%)")

    manejadores, etiquetas = fig.axes[1].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(.5, .075))
    fig.text(.008, .015,
             "El panel (a) muestra el intervalo de Wilson en línea fina y la desviación entre las tres "
             "semillas en banda gruesa. Los paneles (b) y (c) hacen visibles las dos interacciones que el "
             "modelo factorial declara significativas.",
             fontsize=7.2, color="#5b6b7c")

    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (a.out, ext), dpi=200, facecolor="white")
    print("figura ->", a.out + ".png / .pdf")


if __name__ == "__main__":
    main()
