#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ejemplos representativos de errores de reconocimiento de los dos modelos afinados.

Organiza los fallos en las tres categorias que interesan al lector:

  (a) ambos fallan          -> matriculas intrinsecamente dificiles; marcan el techo del metodo
  (b) solo el CRNN falla    -> casos que el LPRNet resuelve
  (c) solo el LPRNet falla  -> casos que el CRNN resuelve

Dentro de cada categoria se eligen ejemplos que cubran distintas posiciones de error y distintas
condiciones de captura, y se resalta el caracter equivocado. La ultima fila muestra ademas la
distribucion de errores por posicion, que revela si el fallo es sistematico o disperso.

Uso:
    python fig_errores.py
"""
import argparse, csv, collections, random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bench_utils import norm_plate
from paired_stats import predecir_ckpt
from redaccion import tapar_caracter, texto_redactado

VERDE, ROJO, GRIS, TINTA = "#137a54", "#c81e2c", "#8a94a0", "#161b21"


def marca(pred, verdad):
    """Devuelve la cadena con el caracter erroneo resaltado mediante $ como delimitador."""
    if len(pred) != len(verdad):
        return pred, list(range(len(pred)))
    malos = [i for i, (a, b) in enumerate(zip(pred, verdad)) if a != b]
    return pred, malos


def formatea(p):
    """Formatea con el cuarto caracter redactado (privacidad)."""
    return texto_redactado(p) if len(p) == 6 else (p or "—")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/rec_labels_gold2_meta.csv")
    ap.add_argument("--lprnet", default="ckpt/lprnet_ft_v2.pt")
    ap.add_argument("--crnn", default="ckpt/crnn_ft_v2.pt")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="../docs/figures/fig4_errores_es.png")
    a = ap.parse_args()

    filas = [r for r in csv.DictReader(open(a.meta, encoding="utf-8")) if Path(r["path"]).exists()]
    rutas = [r["path"] for r in filas]
    verdad = [norm_plate(r["placa"]) for r in filas]

    print("  infiriendo LPRNet afinado ...")
    crudo_l = predecir_ckpt(a.lprnet, rutas)
    print("  infiriendo CRNN afinado ...")
    crudo_c = predecir_ckpt(a.crnn, rutas)

    # Se conserva la salida cruda: cuando el modelo no produce seis caracteres, mostrar la
    # cadena real es mas informativo para el lector que un guion.
    p_lpr = [norm_plate(x) for x in crudo_l]
    p_crn = [norm_plate(x) for x in crudo_c]
    ok_l = [p == v for p, v in zip(p_lpr, verdad)]
    ok_c = [p == v for p, v in zip(p_crn, verdad)]

    grupos = {
        "Ambos fallan": [i for i in range(len(filas)) if not ok_l[i] and not ok_c[i]],
        "Solo falla CRNN": [i for i in range(len(filas)) if ok_l[i] and not ok_c[i]],
        "Solo falla LPRNet": [i for i in range(len(filas)) if not ok_l[i] and ok_c[i]],
    }
    for k, v in grupos.items():
        print(f"  {k:20s} {len(v):4d}")

    def muestra(idx, crudo, normal):
        """Texto a mostrar: la placa formateada, o la salida cruda si no tiene 6 caracteres."""
        if normal[idx]:
            return formatea(normal[idx])
        bruto = crudo[idx].strip()
        return f"«{bruto}»" if bruto else "(vacío)"

    rng = random.Random(a.seed)
    nc = a.cols
    fig = plt.figure(figsize=(nc * 2.45, 3 * 2.05 + 2.0), dpi=300)
    gs = fig.add_gridspec(4, nc, height_ratios=[1, 1, 1, 0.78], hspace=1.15, wspace=0.18)

    for i, (titulo, idxs) in enumerate(grupos.items()):
        def perfil(j):
            """(error_oculto, n_errores): se prefieren fallos pocos y FUERA de la posicion 4,
            que la redaccion cubre y por tanto no seria visible en la figura."""
            p = p_lpr[j] if ("LPRNet" in titulo or "Ambos" in titulo) else p_crn[j]
            if len(p) != len(verdad[j]):
                return (0, 1)
            malos = [i for i, (x, y) in enumerate(zip(p, verdad[j])) if x != y]
            return (1 if malos == [3] else 0, len(malos))
        cand = sorted(idxs, key=lambda j: (perfil(j), rng.random()))[: nc * 3]
        rng.shuffle(cand)
        sel = cand[:nc]
        for j in range(nc):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c9d1d9"); s.set_linewidth(0.6)
            if j >= len(sel):
                ax.set_facecolor("#f2f4f7"); continue
            k = sel[j]
            ax.imshow(tapar_caracter(Image.open(filas[k]["path"]).convert("RGB")))
            luz = {"noche": "noche", "crepusculo": "crep."}.get(filas[k]["luz"], "día")
            col_l = ROJO if not ok_l[k] else VERDE
            col_c = ROJO if not ok_c[k] else VERDE
            ax.set_xlabel(f"real  {formatea(verdad[k])}   ·  {luz}",
                          fontsize=6.4, labelpad=3, color=TINTA, fontweight="bold")
            ax.text(0.02, -0.30, f"LPRNet  {muestra(k, crudo_l, p_lpr)}", transform=ax.transAxes,
                    fontsize=6.2, color=col_l, family="monospace", va="top")
            ax.text(0.02, -0.47, f"CRNN    {muestra(k, crudo_c, p_crn)}", transform=ax.transAxes,
                    fontsize=6.2, color=col_c, family="monospace", va="top")
            if j == 0:
                ax.set_ylabel(f"({chr(97+i)}) {titulo}\nn = {len(idxs)}", fontsize=7.4,
                              rotation=0, ha="right", va="center", labelpad=48, color=TINTA)

    # --- Panel inferior: distribucion de errores por posicion
    ax = fig.add_subplot(gs[3, :])
    pos_l, pos_c = collections.Counter(), collections.Counter()
    for p, v, cont in ((p_lpr, verdad, pos_l), (p_crn, verdad, pos_c)):
        for pred, real in zip(p, v):
            if pred == real:
                continue
            if len(pred) == len(real):
                for idx, (x, y) in enumerate(zip(pred, real)):
                    if x != y:
                        cont[idx + 1] += 1
            else:
                cont[0] += 1        # longitud incorrecta
    posiciones = list(range(1, 7))
    x = np.arange(len(posiciones))
    ax.bar(x - 0.19, [pos_l[p] for p in posiciones], width=0.38, color="#0e7c8b",
           label=f"LPRNet afinado (long. incorrecta: {pos_l[0]})")
    ax.bar(x + 0.19, [pos_c[p] for p in posiciones], width=0.38, color="#b5670a",
           label=f"CRNN afinado (long. incorrecta: {pos_c[0]})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"pos. {p}" for p in posiciones], fontsize=7)
    ax.set_ylabel("Errores", fontsize=7.5)
    ax.set_title("(d) Distribución de los caracteres erróneos por posición en la matrícula",
                 fontsize=8.2, loc="left", weight="bold", pad=6)
    ax.legend(fontsize=6.6, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Errores representativos de los dos reconocedores afinados",
                 fontsize=11, y=0.998, weight="bold")
    fig.text(0.5, -0.005,
             "Recortes reales del conjunto de evaluación. En verde la lectura correcta, en rojo la "
             "errónea. Por privacidad se cubre el primer carácter posterior al guion, tanto en la "
             "imagen como en el texto.\nEl panel (d) agrega todos los errores del conjunto.",
             ha="center", fontsize=6.6, color="#5a6572")

    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    print(f"\nFigura: {dest}")
    print(f"  errores por posición LPRNet: {dict(sorted(pos_l.items()))}")
    print(f"  errores por posición CRNN  : {dict(sorted(pos_c.items()))}")


if __name__ == "__main__":
    main()
