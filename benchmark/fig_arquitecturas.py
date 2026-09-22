#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera los dos diagramas de arquitectura del articulo.

  fig2 -- arquitectura general del sistema, desde el sensor hasta las dos vias de consumo
          (aplicacion de gestion y aplicacion movil), separando lo que ocurre en el borde
          de lo que sale de la red local.
  fig3 -- arquitecturas de los dos reconocedores comparados, con sus dimensiones reales,
          alineadas verticalmente para que la diferencia estructural sea evidente.

Uso:
    python fig_arquitecturas.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

DET, REC, WIN, PERU, TINTA, GRIS, LINEA = ("#0e7c8b", "#b5670a", "#137a54", "#c81e2c",
                                           "#161b21", "#5a6572", "#dde3ea")


def caja(ax, x, y, w, h, texto, sub=None, color=TINTA, relleno="#ffffff", fs=8.0):
    """Caja con titulo y subtitulo. El desplazamiento vertical se calcula a partir del numero
    de lineas de cada parte, para que los titulos de dos lineas no invadan el subtitulo."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=color, facecolor=relleno, zorder=3))
    cx, cy = x + w / 2, y + h / 2
    if not sub:
        ax.text(cx, cy, texto, ha="center", va="center", fontsize=fs, color=TINTA,
                zorder=4, weight="bold")
        return
    n_t, n_s = texto.count("\n") + 1, sub.count("\n") + 1
    alto_t, alto_s = n_t * 0.030, n_s * 0.024
    total = alto_t + alto_s + 0.012
    ax.text(cx, cy + total / 2 - alto_t / 2, texto, ha="center", va="center",
            fontsize=fs, color=TINTA, zorder=4, weight="bold", linespacing=1.15)
    ax.text(cx, cy - total / 2 + alto_s / 2, sub, ha="center", va="center",
            fontsize=fs - 1.6, color=GRIS, zorder=4, linespacing=1.2)


def flecha(ax, x1, y1, x2, y2, color=GRIS, estilo="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=estilo, mutation_scale=11,
                                 linewidth=1.1, color=color, zorder=2,
                                 shrinkA=1, shrinkB=1))


def sistema(dest):
    fig, ax = plt.subplots(figsize=(13.4, 3.9), dpi=300)
    # El limite derecho deja aire para el borde redondeado de la ultima caja, que
    # de otro modo queda recortado por bbox_inches="tight".
    ax.set_xlim(0, 1.02); ax.set_ylim(0.14, 1); ax.axis("off")

    # --- Perímetro del borde: una sola fila, flujo estrictamente de izquierda a derecha.
    ax.add_patch(Rectangle((0.012, 0.545), 0.862, 0.345, linewidth=1.1, edgecolor=DET,
                           facecolor="#0e7c8b0a", linestyle=(0, (5, 3)), zorder=1))
    ax.text(0.018, 0.905, "PROCESAMIENTO EN EL BORDE  ·  red local; el vídeo no sale del perímetro",
            fontsize=7.6, color=DET, weight="bold", va="bottom")

    y, h, w = 0.605, 0.225, 0.104
    etapas = [
        (0.024, "Cámara IP", "8 MP varifocal\n2.50 m · <45°", DET),
        (0.160, "Flujo RTSP", "1280×720\n10 fps, CBR", DET),
        (0.296, "Detector", "YOLO\ncaja de placa", DET),
        (0.432, "Recorte", "ceñido\na la caja", DET),
        (0.568, "OCR + corrector", "PaddleOCR ·\nreglas por posición", REC),
    ]
    for x, t, s, c in etapas:
        caja(ax, x, y, w, h, t, s, c, "#ffffff", fs=7.6)
    for i in range(len(etapas) - 1):
        flecha(ax, etapas[i][0] + w, y + h / 2, etapas[i + 1][0], y + h / 2)

    # --- Filtros y salida
    caja(ax, 0.760, y, 0.108, h, "Filtros", "deduplicación\nformato · área", REC)
    flecha(ax, 0.672, y + h / 2, 0.760, y + h / 2)

    caja(ax, 0.760, 0.235, 0.108, h, "API REST", "solo el registro\nestructurado", WIN, "#137a5410")
    flecha(ax, 0.814, y, 0.814, 0.235 + h, WIN)

    caja(ax, 0.895, 0.235, 0.098, h, "Registro", "matrícula ·\nmetadatos", TINTA)
    flecha(ax, 0.868, 0.235 + h / 2, 0.895, 0.235 + h / 2)

    # --- Rama de destilación: el OCR corregido genera las etiquetas con las que se destila
    # el reconocedor compacto, que vuelve al borde a sustituirlo.
    caja(ax, 0.190, 0.235, 0.160, h, "Etiquetas silver", "579 292 recortes\npara destilación",
         REC, "#b5670a10")
    flecha(ax, 0.600, y, 0.350, 0.235 + h / 2, REC)

    caja(ax, 0.420, 0.235, 0.150, h, "Reconocedor compacto", "LPRNet 0.97 M\nONNX, 3.7 MB",
         WIN, "#137a5410", fs=7.4)
    flecha(ax, 0.350, 0.235 + h / 2, 0.420, 0.235 + h / 2, REC)

    # Realimentación al borde: el estudiante destilado reemplaza al maestro en inferencia.
    flecha(ax, 0.570, 0.300, 0.660, 0.300, WIN)
    flecha(ax, 0.660, 0.300, 0.660, y, WIN)
    ax.text(0.615, 0.500, "despliegue\nen el borde", fontsize=6.6, color=WIN,
            ha="center", va="center", linespacing=1.3)

    ax.text(0.5, 0.075,
            "El flujo de vídeo permanece en la red local; solo el registro estructurado abandona el "
            "borde. La salida del OCR corregido constituye las etiquetas con las\nque se destila el "
            "reconocedor compacto, que regresa al borde para sustituir al maestro en inferencia sin "
            "requerir post-proceso.",
            ha="center", fontsize=7.2, color=GRIS)
    ax.set_title("Arquitectura general del sistema de reconocimiento de matrículas propuesto",
                 fontsize=11.5, weight="bold", pad=10)
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    print("Figura:", dest)


def modelos(dest):
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 4.4), dpi=300)

    # Etiquetas breves: el detalle dimensional va debajo de cada bloque.
    capas_lpr = [
        ("Entrada", "3×24×94", GRIS),
        ("Conv 64", "64×24×94", DET),
        ("Pool 2×2", "64×12×47", "#9fb3bd"),
        ("Conv 128", "128×12×47", DET),
        ("Pool 2×2", "128×6×23", "#9fb3bd"),
        ("Conv 256", "256×6×23", DET),
        ("Pool 2,1", "256×3×23", "#9fb3bd"),
        ("Conv 256", "256×3×23", DET),
        ("Pool 3,1", "256×1×23", "#9fb3bd"),
        ("Dropout", "0.5", "#c9d1d9"),
        ("Conv 1×1", "37×1×23", WIN),
        ("CTC", "T=23", PERU),
    ]
    capas_crnn = [
        ("Entrada", "3×32×100", GRIS),
        ("Conv 64", "64×16×50", REC),
        ("Conv 128", "128×8×25", REC),
        ("Conv 256×2", "256×8×25", REC),
        ("Pool 2,1", "256×4×25", "#d9b98a"),
        ("Conv 512×2", "512×4×25", REC),
        ("Pool 2,1", "512×2×25", "#d9b98a"),
        ("Conv 512", "512×1×24", REC),
        ("BiLSTM ×2", "24×512", PERU),
        ("Lineal", "24×37", WIN),
        ("CTC", "T=24", PERU),
    ]

    for ax, capas, titulo, sub in (
            (axes[0], capas_lpr, "LPRNet — 0.97 M parámetros",
             "puramente convolucional; la altura colapsa a 1 y el ancho actúa como eje temporal"),
            (axes[1], capas_crnn, "CRNN — 8.73 M parámetros",
             "convolucional + memoria secuencial bidireccional sobre el eje temporal")):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        n = len(capas)
        hueco = 0.955 / n
        w = hueco * 0.76          # el 24 % restante deja aire para la flecha
        for i, (nombre, dim, color) in enumerate(capas):
            x = 0.022 + i * hueco
            caja(ax, x, 0.34, w, 0.30, nombre, None, color, "#ffffff", fs=6.8)
            ax.text(x + w / 2, 0.285, dim, ha="center", va="top", fontsize=5.9,
                    color=GRIS, family="monospace")
            if i < n - 1:
                flecha(ax, x + w + 0.004, 0.49, x + hueco - 0.004, 0.49)
        ax.set_title(titulo, fontsize=10, weight="bold", loc="left", pad=1)
        ax.text(0.0, 0.79, sub, fontsize=7.2, color=GRIS, transform=ax.transAxes)

    fig.suptitle("Arquitecturas de los reconocedores comparados", fontsize=11.5, weight="bold",
                 y=0.995)
    fig.text(0.5, 0.005,
             "Ambos se entrenan con pérdida CTC sobre 37 clases (10 dígitos, 26 letras y el "
             "símbolo en blanco) y no requieren segmentar caracteres.",
             ha="center", fontsize=7.0, color=GRIS)
    fig.subplots_adjust(hspace=0.42, top=0.88, bottom=0.08)
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    print("Figura:", dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="../docs/figures")
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    sistema(d / "fig2_arquitectura_sistema_es.png")
    modelos(d / "fig3_arquitectura_modelos_es.png")


if __name__ == "__main__":
    main()
