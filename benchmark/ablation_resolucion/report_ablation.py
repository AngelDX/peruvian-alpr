#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabla LaTeX, figura y contrastes pareados de la ablacion de resolucion.

Uso (desde benchmark/):
    python ablation_resolucion/report_ablation.py
    python ablation_resolucion/report_ablation.py --json ablation_resolucion/resultados/ablacion_resolucion_s0.json

Genera en resultados/:
    tabla_resolucion.tex     tabla lista para pegar en el manuscrito
    fig_resolucion.png/.pdf  exactitud frente a resolucion
    contrastes.txt           McNemar exacta con correccion de Holm
"""
import os
import sys
import json
import glob
import argparse
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)


def mcnemar_exacta(a_pred, b_pred, truth):
    """p bilateral de la binomial exacta sobre los discordantes."""
    b10 = sum(1 for p, q, t in zip(a_pred, b_pred, truth) if (p == t) and (q != t))
    b01 = sum(1 for p, q, t in zip(a_pred, b_pred, truth) if (p != t) and (q == t))
    n = b10 + b01
    if n == 0:
        return 1.0, b10, b01
    k = min(b10, b01)
    cola = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cola), b10, b01


def holm(pvals):
    orden = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, ajust, previo = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(orden):
        v = min(1.0, (m - rank) * pvals[i])
        previo = max(previo, v)
        ajust[i] = previo
    return ajust


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def cargar(patron):
    filas = []
    for f in sorted(glob.glob(patron)):
        with open(f, encoding="utf-8") as fh:
            filas.extend(json.load(fh))
    if not filas:
        raise SystemExit("no encontre resultados en %s" % patron)
    return filas


def promediar(filas):
    """Si hay varias semillas, promedia y guarda la desviacion."""
    grupos = {}
    for r in filas:
        grupos.setdefault((r["modelo"], r["alto"], r["ancho"]), []).append(r)
    out = []
    for (m, h, w), rs in grupos.items():
        base = dict(rs[0])
        ex = [x["gold_exact"] for x in rs]
        base["gold_exact"] = sum(ex) / len(ex)
        base["sd"] = (sum((e - base["gold_exact"]) ** 2 for e in ex) / (len(ex) - 1)) ** 0.5 if len(ex) > 1 else 0.0
        base["n_semillas"] = len(rs)
        base["cer"] = sum(x["gold_cer"] for x in rs) / len(rs)
        out.append(base)
    return out


def tabla_latex(filas, destino):
    filas = sorted(filas, key=lambda r: (r["modelo"], r["alto"]))
    L = []
    L.append("% Generado por ablation_resolucion/report_ablation.py")
    L.append(r"\begin{table}[H]")
    L.append(r"\caption{Ablación de la resolución de normalización de la ROI. El aspecto nativo")
    L.append(r"mediano de la matrícula peruana es 1.98. Coincidencia exacta sobre \emph{Gold}-B")
    L.append(r"($n = 1\,130$).\label{tab:resolucion}}")
    L.append(r"\begin{tabularx}{\textwidth}{lXrrrrr}")
    L.append(r"\toprule")
    L.append(r"\textbf{Modelo} & \textbf{Entrada} & \textbf{Aspecto} & \textbf{$T$} & "
             r"\textbf{Exacta} & \textbf{CER} & \textbf{ms/img} \\")
    L.append(r"\midrule")
    modelo_previo = None
    mejor = {}
    for r in filas:
        mejor.setdefault(r["modelo"], max(
            (x for x in filas if x["modelo"] == r["modelo"]), key=lambda y: y["gold_exact"]))
    for r in filas:
        if modelo_previo is not None and r["modelo"] != modelo_previo:
            L.append(r"\midrule")
        modelo_previo = r["modelo"]
        ex = "%.2f\\,\\%%" % (100 * r["gold_exact"])
        if r is mejor[r["modelo"]] or (r["alto"], r["ancho"]) == (mejor[r["modelo"]]["alto"], mejor[r["modelo"]]["ancho"]):
            ex = r"\textbf{%s}" % ex
        L.append("%s & $%d\\times%d$ (%s) & %.2f & %d & %s & %.4f & %.2f \\\\"
                 % (r["modelo"].upper(), r["alto"], r["ancho"], r.get("etiqueta", ""),
                    r["aspecto"], r["T"], ex, r.get("cer", r["gold_cer"]), r["ms_img"]))
    L.append(r"\bottomrule")
    L.append(r"\end{tabularx}")
    L.append(r"\end{table}")
    txt = "\n".join(L) + "\n"
    with open(destino, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def contrastes(crudas, destino):
    """McNemar pareada de cada resolucion contra la linea base del modelo."""
    lineas = []
    for modelo in sorted({r["modelo"] for r in crudas}):
        rs = [r for r in crudas if r["modelo"] == modelo and r.get("semilla", 0) == 0]
        if not rs:
            continue
        base_wh = (24, 94) if modelo == "lprnet" else (32, 100)
        base = next((r for r in rs if (r["alto"], r["ancho"]) == base_wh), None)
        if base is None:
            lineas.append("%s: sin linea base %dx%d, omito los contrastes"
                          % (modelo, base_wh[0], base_wh[1]))
            continue
        otros = [r for r in rs if r is not base]
        ps, det = [], []
        for r in otros:
            p, b10, b01 = mcnemar_exacta(r["pred"], base["pred"], base["truth"])
            ps.append(p)
            det.append((r, b10, b01))
        aj = holm(ps) if ps else []
        lineas.append("== %s : cada resolución frente a la línea base %dx%d ==" % (modelo.upper(), *base_wh))
        n = len(base["truth"])
        lo, hi = wilson(round(base["gold_exact"] * n), n)
        lineas.append("   línea base  exacta %.4f  IC95 [%.4f, %.4f]" % (base["gold_exact"], lo, hi))
        for (r, b10, b01), p, pa in zip(det, ps, aj):
            d = 100 * (r["gold_exact"] - base["gold_exact"])
            sig = "significativo" if pa < 0.05 else "no significativo"
            lineas.append("   %3dx%-4d  %+6.2f pp   ganadas %3d / perdidas %3d   p=%.3g  p_Holm=%.3g  %s"
                          % (r["alto"], r["ancho"], d, b10, b01, p, pa, sig))
        lineas.append("")
    txt = "\n".join(lineas) + "\n"
    with open(destino, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def figura(filas, destino_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible, omito la figura")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for modelo, marca in (("lprnet", "o"), ("crnn", "s")):
        rs = sorted([r for r in filas if r["modelo"] == modelo], key=lambda r: r["alto"] * r["ancho"])
        if not rs:
            continue
        px = [r["alto"] * r["ancho"] / 1000 for r in rs]
        ex = [100 * r["gold_exact"] for r in rs]
        sd = [100 * r.get("sd", 0.0) for r in rs]
        ax[0].errorbar(px, ex, yerr=sd, marker=marca, capsize=3, label=modelo.upper())
        for r, x, y in zip(rs, px, ex):
            ax[0].annotate("%dx%d" % (r["alto"], r["ancho"]), (x, y),
                           textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
        ax[1].scatter([r["aspecto"] for r in rs], ex, marker=marca, label=modelo.upper())
    ax[0].set_xlabel("píxeles de entrada (miles)")
    ax[0].set_ylabel("coincidencia exacta en Gold-B (%)")
    ax[0].grid(alpha=.3)
    ax[0].legend()
    ax[1].axvline(1.98, ls="--", c="k", lw=1)
    ax[1].annotate("aspecto nativo 1.98", (1.98, ax[1].get_ylim()[0]),
                   rotation=90, fontsize=8, va="bottom", ha="right")
    ax[1].set_xlabel("aspecto de la entrada (ancho/alto)")
    ax[1].set_ylabel("coincidencia exacta (%)")
    ax[1].grid(alpha=.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(destino_png, dpi=200)
    fig.savefig(destino_png.replace(".png", ".pdf"))
    print("figura ->", destino_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "resultados", "ablacion_resolucion_s*.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "resultados"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    crudas = cargar(a.json)
    filas = promediar(crudas)

    print(tabla_latex(filas, os.path.join(a.out, "tabla_resolucion.tex")))
    print(contrastes(crudas, os.path.join(a.out, "contrastes.txt")))
    figura(filas, os.path.join(a.out, "fig_resolucion.png"))
    print("salidas en", a.out)


if __name__ == "__main__":
    main()
