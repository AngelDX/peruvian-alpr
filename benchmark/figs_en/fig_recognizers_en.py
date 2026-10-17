#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3: architectures of the two recognizers, at the unified 32x100 input.

Replaces the earlier version, which still drew LPRNet at 24x94 and therefore
contradicted the unified-input protocol reported in the manuscript. Tensor
shapes follow models.py exactly.

Usage (from benchmark/figs_en):
    python fig_recognizers_en.py
"""
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Rutas resueltas desde la ubicacion del script, no desde el directorio de trabajo.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_AQUI)
_TEX = os.path.abspath(os.path.join(_BENCH, "..", "docs", "paper_sensors", "tex"))


def _b(*p):
    return os.path.join(_BENCH, *p)


def _t(nombre):
    return os.path.join(_TEX, nombre)



DET, REC, WIN, PERU, TINTA, GRIS = ("#0e7c8b", "#b5670a", "#137a54",
                                    "#c81e2c", "#161b21", "#5a6572")


def caja(ax, x, y, w, h, texto, color, fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, edgecolor=color, facecolor="#ffffff",
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            color=TINTA, zorder=3)


def flecha(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9, color=GRIS, zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_t("fig3_recognizers_en.png"))
    a = ap.parse_args()

    # Formas exactas de models.py con entrada 3x32x100.
    capas_lpr = [
        ("Input", "3×32×100", GRIS),
        ("Conv 64", "64×32×100", DET),
        ("Pool 2×2", "64×16×50", "#9fb3bd"),
        ("Conv 128", "128×16×50", DET),
        ("Pool 2×2", "128×8×25", "#9fb3bd"),
        ("Conv 256", "256×8×25", DET),
        ("Pool 2,1", "256×4×25", "#9fb3bd"),
        ("Conv 256", "256×4×25", DET),
        ("AdaptPool", "256×1×25", "#9fb3bd"),
        ("Dropout", "0.5", "#c9d1d9"),
        ("Conv 1×1", "37×1×25", WIN),
        ("CTC", "T = 25", PERU),
    ]
    capas_crnn = [
        ("Input", "3×32×100", GRIS),
        ("Conv 64", "64×16×50", REC),
        ("Conv 128", "128×8×25", REC),
        ("Conv 256×2", "256×8×25", REC),
        ("Pool 2,1", "256×4×25", "#d9b98a"),
        ("Conv 512×2", "512×4×25", REC),
        ("Pool 2,1", "512×2×25", "#d9b98a"),
        ("Conv 512", "512×1×24", REC),
        ("BiLSTM ×2", "24×512", PERU),
        ("Linear", "24×37", WIN),
        ("CTC", "T = 24", PERU),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 4.4), dpi=300)
    for ax, capas, titulo, sub in (
            (axes[0], capas_lpr, "LPRNet — 0.97 M parameters",
             "purely convolutional; feature-map height collapses to 1 and width acts as the temporal axis"),
            (axes[1], capas_crnn, "CRNN — 8.73 M parameters",
             "convolutional stack plus bidirectional sequential memory over the temporal axis")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        n = len(capas)
        hueco = 0.955 / n
        w = hueco * 0.76
        for i, (nombre, dim, color) in enumerate(capas):
            x = 0.022 + i * hueco
            caja(ax, x, 0.34, w, 0.30, nombre, color)
            ax.text(x + w / 2, 0.285, dim, ha="center", va="top", fontsize=5.9,
                    color=GRIS, family="monospace")
            if i < n - 1:
                flecha(ax, x + w + 0.004, 0.49, x + hueco - 0.004, 0.49)
        ax.set_title(titulo, fontsize=10, weight="bold", loc="left", pad=1)
        ax.text(0.0, 0.79, sub, fontsize=7.2, color=GRIS, transform=ax.transAxes)

    fig.suptitle("Architectures of the two recognizers compared, at the unified 32 × 100 input",
                 fontsize=11.5, weight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "Both are trained with CTC loss over 37 classes (10 digits, 26 letters and the blank "
             "symbol) and require no character segmentation.",
             ha="center", fontsize=7.0, color=GRIS)
    fig.subplots_adjust(hspace=0.42, top=0.88, bottom=0.08)

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(dest.with_suffix("." + ext), bbox_inches="tight", facecolor="white", dpi=300)
    print("figure ->", dest.with_suffix(".png"))


if __name__ == "__main__":
    main()
