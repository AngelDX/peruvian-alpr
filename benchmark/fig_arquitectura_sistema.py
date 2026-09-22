#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagrama de arquitectura del sistema (Figura 1 del manuscrito).

Estilo de cajas y flechas. Dos ramas a partir del recorte normalizado:
el maestro, que etiqueta el corpus fuera de linea, y el estudiante, que es
lo que finalmente corre en el servidor de borde.

Uso (desde benchmark/):
    python fig_arquitectura_sistema.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Paleta
NARANJA = "#E07B39"
NARANJA_C = "#F7C9A8"
AZUL = "#3F72AF"
AZUL_C = "#C5D9F1"
VERDE = "#4F8A4F"
VERDE_C = "#C6E0C6"
GRIS = "#5B6B7C"
TINTA = "#16202B"


def caja(ax, x, y, w, h, titulo, sub=None, borde=GRIS, relleno="white",
         fs=10, fs_sub=8.5, negrita=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                linewidth=1.6, edgecolor=borde, facecolor=relleno,
                                zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * .63, titulo, ha="center", va="center",
                fontsize=fs, color=TINTA, fontweight="bold" if negrita else "normal",
                zorder=3)
        ax.text(x + w / 2, y + h * .27, sub, ha="center", va="center",
                fontsize=fs_sub, color=GRIS, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, titulo, ha="center", va="center",
                fontsize=fs, color=TINTA, fontweight="bold" if negrita else "normal",
                zorder=3, linespacing=1.35)


def flecha(ax, p0, p1, color=TINTA, estilo="-", rad=0.0, lw=1.7, texto=None,
           dx=0, dy=.16, fs=8.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, linestyle=estilo,
                                 connectionstyle="arc3,rad=%.2f" % rad,
                                 shrinkA=2, shrinkB=2, zorder=4))
    if texto:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, texto,
                ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../docs/paper_sensors/tex/fig2_arquitectura_sistema_es")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    H = .92                                   # alto estandar de caja

    # ---------------------------------------------------------------- fila 1
    y1 = 5.35
    caja(ax, .25, y1, 2.05, H, "Fotograma", "1280 × 720 px\ncámara IP fija", GRIS, "white")
    caja(ax, 2.95, y1, 2.35, H, "YOLOv8n", "detector de placa\n1.1 ms/img", NARANJA, NARANJA_C)
    caja(ax, 5.95, y1, 2.75, H, "Recorte de la placa",
         "tamaño nativo\nmediana 51 × 101 px", AZUL, AZUL_C)
    caja(ax, 9.35, y1, 2.75, H, "Filtros y registro",
         "formato LNN-DDD · deduplicación\nmatrícula + hora vía API REST", GRIS, "white")

    flecha(ax, (2.30, y1 + H / 2), (2.95, y1 + H / 2))
    flecha(ax, (5.30, y1 + H / 2), (5.95, y1 + H / 2))
    flecha(ax, (8.70, y1 + H / 2), (9.35, y1 + H / 2))

    # ------------------------------------------------- fila 2: maestro
    y2 = 3.30
    ax.add_patch(Rectangle((.15, y2 - .52), 11.95, H + .95, linewidth=1.1,
                           edgecolor=AZUL, facecolor="#F2F7FD", linestyle=(0, (5, 3)),
                           zorder=1))
    ax.text(.35, y2 + H + .24, "MAESTRO   ·   etiquetado del corpus, fuera de línea",
            fontsize=8.5, color=AZUL, fontweight="bold", zorder=3)

    caja(ax, .45, y2, 2.55, H, "PaddleOCR PP-OCRv4",
         "sobre el recorte nativo", AZUL, AZUL_C)
    caja(ax, 3.65, y2, 2.85, H, "Corrector determinista", "por posición · O/0 e I/1", AZUL, AZUL_C)
    caja(ax, 7.15, y2, 2.55, H, "Etiquetas $silver$", "579 292 recortes", AZUL, AZUL_C)

    flecha(ax, (3.00, y2 + H / 2), (3.65, y2 + H / 2), AZUL)
    flecha(ax, (6.50, y2 + H / 2), (7.15, y2 + H / 2), AZUL)

    # ---------------------------------------------- fila 3: estudiante
    y3 = 1.00
    ax.add_patch(Rectangle((.15, y3 - .52), 11.95, H + .95, linewidth=1.1,
                           edgecolor=VERDE, facecolor="#F3F9F3", linestyle=(0, (5, 3)),
                           zorder=1))
    ax.text(.35, y3 + H + .24,
            "ESTUDIANTE   ·   inferencia en el servidor de borde, sobre el mismo recorte",
            fontsize=8.5, color=VERDE, fontweight="bold", zorder=3)

    # Fila invertida: el reconocedor queda bajo las etiquetas silver, de modo que
    # la destilacion es una vertical corta en lugar de una diagonal que cruza todo.
    # La normalizacion vive aqui, no en la cadena comun: solo LPRNet y CRNN la
    # necesitan. El maestro lee el recorte nativo con su propio preprocesado.
    caja(ax, 10.05, y3, 1.95, H, "Normalización", "32 × 100 px\nRGB", VERDE, VERDE_C)
    caja(ax, 7.15, y3, 2.55, H, "Extracción de rasgos",
         "LPRNet 0.97 M\no CRNN 8.73 M", VERDE, VERDE_C)
    caja(ax, 3.65, y3, 2.85, H, "Decodificación CTC",
         "37 clases · sin segmentar", VERDE, VERDE_C)
    caja(ax, .45, y3, 2.55, H, "Texto predicho", "Z7C-0\u25cf9", TINTA, "white")

    flecha(ax, (10.05, y3 + H / 2), (9.70, y3 + H / 2), VERDE)
    flecha(ax, (7.15, y3 + H / 2), (6.50, y3 + H / 2), VERDE)
    flecha(ax, (3.65, y3 + H / 2), (3.00, y3 + H / 2), VERDE)

    # ajuste fino, sin caja propia para no cargar la fila
    ax.add_patch(FancyArrowPatch((8.42, y3 - .40), (8.42, y3), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.7, color=VERDE, zorder=4))
    ax.text(8.60, y3 - .30, "ajuste fino · 1 850 recortes verificados", fontsize=8.5,
            color=VERDE, fontweight="bold", zorder=5, ha="left", va="center")

    # ------------------------------------------------------------ enlaces
    # el recorte baja al maestro por la izquierda, en codo ortogonal
    # Codo ortogonal en tres tramos: bajar, ir a la izquierda y bajar. Se dibuja a
    # mano porque connectionstyle="angle" no admite angulos opuestos.
    yb = y1 - .58
    ax.plot([7.32, 7.32, 1.72], [y1, yb, yb], color=AZUL, linewidth=1.9,
            solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.add_patch(FancyArrowPatch((1.72, yb), (1.72, y2 + H), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.9, color=AZUL,
                                 shrinkA=0, shrinkB=2, zorder=4))
    ax.text(4.55, yb, "recorte nativo", fontsize=8.5, color=AZUL,
            fontweight="bold", zorder=5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none"))
    # el mismo recorte baja tambien al estudiante, por el corredor de la derecha
    ax.plot([7.32, 11.02, 11.02], [yb, yb, y3 + H + .30], color=AZUL, linewidth=1.9,
            solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.add_patch(FancyArrowPatch((11.02, y3 + H + .30), (11.02, y3 + H),
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.9,
                                 color=AZUL, shrinkA=0, shrinkB=2, zorder=4))
    # destilacion: vertical corta
    flecha(ax, (8.42, y2), (8.42, y3 + H), VERDE, lw=2.2)
    ax.text(8.60, (y2 + y3 + H) / 2, "destilación con\nsupervisión débil", fontsize=8.5,
            color=VERDE, fontweight="bold", zorder=5, ha="left", va="center",
            linespacing=1.3)

    fig.text(.012, .022,
             "El flujo de vídeo no abandona la red local: solo el registro estructurado sale del borde. "
             "El maestro etiqueta el corpus una vez; el estudiante lo sustituye en producción y no necesita el corrector.",
             fontsize=8, color=GRIS)

    fig.subplots_adjust(left=.005, right=.995, top=.99, bottom=.055)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (a.out, ext), dpi=220)
    print("figura ->", a.out + ".png / .pdf")


if __name__ == "__main__":
    main()
