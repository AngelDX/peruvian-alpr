#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de caja y bigotes para los contrastes post-hoc sobre Gold-B.

La tabla post-hoc da un punto y un valor p por comparacion. Eso basta para decidir
pero no para interpretar: no se ve cuanta incertidumbre hay detras de cada
diferencia ni como se solapan entre si. Esta figura dibuja la distribucion
completa por bootstrap apareado, remuestreando matriculas (no predicciones) para
preservar la correlacion entre sistemas, que es la misma convencion del analisis
estadistico del articulo.

Dos paneles:
  (a) Exactitud de cada sistema. Una caja por sistema.
  (b) Las k(k-1)/2 diferencias apareadas, ordenadas por mediana, con la linea del
      cero marcada y la significacion tras Holm anotada al margen.

Entrada: results/matriz_gold2.csv, que genera analisis_inferencial.py.

Uso (desde benchmark/):
    python analisis_inferencial.py            # si aun no existe la matriz
    python fig_posthoc_cajas.py
"""
import os
import csv
import argparse
import itertools
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rutas resueltas desde la ubicacion del script, no desde el directorio de trabajo.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_AQUI)
_TEX = os.path.abspath(os.path.join(_BENCH, "..", "docs", "paper_sensors", "tex"))


def _b(*p):
    return os.path.join(_BENCH, *p)


def _t(nombre):
    return os.path.join(_TEX, nombre)



ORDEN = ["PaddleOCR crudo", "PaddleOCR + corrector", "CRNN", "LPRNet",
         "CRNN afinado", "LPRNet afinado",
         "PaddleOCR+corr", "LPRNet-ft"]   # los dos ultimos, nombres antiguos
BONITO = {"PaddleOCR crudo": "PaddleOCR\nraw", "PaddleOCR + corrector": "PaddleOCR\n+ corrector",
          "PaddleOCR+corr": "PaddleOCR\n+ corrector", "CRNN": "CRNN", "LPRNet": "LPRNet",
          "CRNN afinado": "CRNN\nfine-tuned", "LPRNet afinado": "LPRNet\nfine-tuned",
          "LPRNet-ft": "LPRNet\nfine-tuned"}
# Version compacta para las etiquetas del panel de diferencias, que de otro modo
# se desbordan sobre el panel izquierdo.
CORTO = {"PaddleOCR crudo": "raw OCR", "PaddleOCR + corrector": "OCR+corr",
         "PaddleOCR+corr": "OCR+corr", "CRNN": "CRNN", "LPRNet": "LPRNet",
         "CRNN afinado": "CRNN-ft", "LPRNet afinado": "LPRNet-ft", "LPRNet-ft": "LPRNet-ft"}


def leer(ruta):
    with open(ruta, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    if not filas:
        raise SystemExit("empty matrix: %s" % ruta)
    cols = [c for c in filas[0] if c in ORDEN] or \
           [c for c in filas[0] if c not in ("placa", "verdad") and not c.startswith("pred_")]
    cols = [c for c in ORDEN if c in cols] + [c for c in cols if c not in ORDEN]
    X = np.array([[int(f[c]) for c in cols] for f in filas], dtype=int)
    return cols, X


def mcnemar(a, b):
    b10 = int(np.sum((a == 1) & (b == 0)))
    b01 = int(np.sum((a == 0) & (b == 1)))
    n = b10 + b01
    if not n:
        return 1.0, b10, b01
    k = min(b10, b01)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), b10, b01


def holm(p):
    orden = sorted(range(len(p)), key=lambda i: p[i])
    m, aj, prev = len(p), [0.0] * len(p), 0.0
    for r, i in enumerate(orden):
        prev = max(prev, min(1.0, (m - r) * p[i]))
        aj[i] = prev
    return aj


def estrellas(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


def _orient(vertical):
    """boxplot cambio 'vert' por 'orientation' en Matplotlib 3.11."""
    from matplotlib import __version__ as mplv
    partes = tuple(int(x) for x in mplv.split(".")[:2])
    if partes >= (3, 11):
        return {"orientation": "vertical" if vertical else "horizontal"}
    return {"vert": vertical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matriz", default=_b("results", "matriz_gold2_32x100.csv"))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--figsize", default="13,7.2")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--out", default=_t("fig6_posthoc_boxes_en"))
    a = ap.parse_args()

    if not os.path.exists(a.matriz):
        raise SystemExit("falta %s. Generalo con:  python analisis_inferencial.py" % a.matriz)

    cols, X = leer(a.matriz)
    n, k = X.shape
    print("matrix: %d plates x %d systems" % (n, k))

    rng = np.random.default_rng(a.semilla)
    idx = rng.integers(0, n, size=(a.reps, n))
    # Remuestreo de matriculas: la misma seleccion para todos los sistemas, que es
    # lo que preserva el apareamiento.
    B = X[idx].mean(axis=1)                       # (reps, k)

    pares = list(itertools.combinations(range(k), 2))
    difs, etiquetas, ps, bb = [], [], [], []
    for i, j in pares:
        d = B[:, i] - B[:, j]
        if d.mean() < 0:
            i, j = j, i
            d = -d
        difs.append(100 * d)
        etiquetas.append("%s  −  %s" % (CORTO.get(cols[i], cols[i]), CORTO.get(cols[j], cols[j])))
        p, b10, b01 = mcnemar(X[:, i], X[:, j])
        ps.append(p)
        bb.append((b10, b01))
    aj = holm(ps)

    orden = sorted(range(len(difs)), key=lambda t: np.median(difs[t]))
    difs = [difs[t] for t in orden]
    etiquetas = [etiquetas[t] for t in orden]
    aj = [aj[t] for t in orden]
    bb = [bb[t] for t in orden]

    fig = plt.figure(figsize=tuple(float(x) for x in a.figsize.split(",")))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=.42,
                          left=.065, right=.985, top=.9, bottom=.16)

    # ---- (a) exactitud por sistema
    ax = fig.add_subplot(gs[0])
    orden_s = np.argsort(B.mean(axis=0))
    cajas = ax.boxplot([100 * B[:, j] for j in orden_s], patch_artist=True,
                       whis=(2.5, 97.5), showfliers=False, widths=.6, **_orient(True))
    for c in cajas["boxes"]:
        c.set(facecolor="#cfe0f5", edgecolor="#26527d", linewidth=1.1)
    for w in cajas["whiskers"] + cajas["caps"]:
        w.set(color="#26527d", linewidth=1.1)
    for m in cajas["medians"]:
        m.set(color="#b3261e", linewidth=1.8)
    ax.set_xticklabels([BONITO.get(cols[j], cols[j]) for j in orden_s], fontsize=8)
    ax.set_ylabel("exact match (%)")
    ax.set_title("(a) Accuracy by system", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.3)
    alto = ax.get_ylim()[1] - ax.get_ylim()[0]
    for t, j in enumerate(orden_s, start=1):
        v = 100 * B[:, j]
        ax.annotate("%.1f" % v.mean(), (t, np.percentile(v, 97.5)),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=7.5, color="#26527d")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + alto * .05)

    # ---- (b) diferencias apareadas
    ax = fig.add_subplot(gs[1])
    cajas = ax.boxplot(difs, patch_artist=True, whis=(2.5, 97.5),
                       showfliers=False, widths=.62, **_orient(False))
    for c, p in zip(cajas["boxes"], aj):
        c.set(facecolor="#cfe8d5" if p < .05 else "#eeeeee",
              edgecolor="#1a7f4b" if p < .05 else "#8a8a8a", linewidth=1.1)
    for w in cajas["whiskers"] + cajas["caps"]:
        w.set(color="#5b6b7c", linewidth=1.0)
    for m in cajas["medians"]:
        m.set(color="#b3261e", linewidth=1.8)
    ax.axvline(0, color="#16202b", lw=1.2, ls="--")
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set_xlabel("difference in exact match (percentage points)")
    ax.set_title("(b) Paired differences, bootstrap of %s replicates" % f"{a.reps:,}",
                 fontsize=10, loc="left")
    ax.grid(axis="x", alpha=.3)
    lim = ax.get_xlim()
    ax.set_xlim(lim[0], lim[1] + (lim[1] - lim[0]) * .17)
    for t, (d, p, (b10, b01)) in enumerate(zip(difs, aj, bb), start=1):
        ax.annotate("%s  %d/%d" % (estrellas(p), b10, b01),
                    (ax.get_xlim()[1], t), xytext=(-4, -3), textcoords="offset points",
                    ha="right", fontsize=7.5,
                    color="#1a7f4b" if p < .05 else "#8a6d1f")

    fig.text(.008, .035,
             "Boxes: quartiles; whiskers: 2.5th and 97.5th bootstrap percentiles, resampled over plates. "
             "Red line: median. Annotation: significance after Holm and discordant cases favoring each system. "
             # La leyenda lista solo los niveles que aparecen en la figura:
             # produccion hizo notar que no hay ningun ** en estos datos.
             + "  ".join(l for s_, l in (("***", "*** p < 0.001"), ("**", "** p < 0.01"),
                                         ("*", "* p < 0.05"), ("n.s.", "n.s. not significant"))
                         if s_ in {estrellas(p) for p in aj}),
             fontsize=7.2, color="#5b6b7c")
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (a.out, ext), dpi=300)
    print("figura ->", a.out + ".png / .pdf")

    print("\n%-46s %8s %10s %10s" % ("comparison", "Δ (pp)", "p Holm", "b10/b01"))
    for d, e, p, (b10, b01) in zip(difs[::-1], etiquetas[::-1], aj[::-1], bb[::-1]):
        print("%-46s %8.2f %10.2e %6d/%-6d" % (e, np.median(d), p, b10, b01))


if __name__ == "__main__":
    main()
