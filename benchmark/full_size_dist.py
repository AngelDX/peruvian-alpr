#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribucion de tamano de placa sobre TODO el dataset (figura Data Records).

Extrae ancho/alto (px) del nombre de cada crop en crops/{train,val,test}
(sin leer las imagenes). Soporta dos formatos: "(H, W, 3)" y "WxH".
Genera fig_plate_size_distribution.{png,pdf} y full_size_distribution.json.
"""
import os, re, json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RX_TUPLE = re.compile(r"\((\d+),\s*(\d+),\s*\d+\)")   # (H, W, C)
RX_WXH   = re.compile(r"_(\d+)x(\d+)[_.]")             # _WxH_


def parse_wh(fn):
    m = RX_TUPLE.search(fn)
    if m:
        h, w = int(m.group(1)), int(m.group(2))
        return w, h
    m = RX_WXH.search(fn)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        return w, h
    return None


def collect(dataset):
    W, H, split_of = [], [], []
    counts = {}
    for split in ("train", "val", "test"):
        d = os.path.join(dataset, "crops", split)
        if not os.path.isdir(d):
            continue
        n = bad = 0
        with os.scandir(d) as it:
            for e in it:
                if not e.name.lower().endswith(".jpg"):
                    continue
                wh = parse_wh(e.name)
                if wh is None:
                    bad += 1; continue
                W.append(wh[0]); H.append(wh[1]); split_of.append(split)
                n += 1
        counts[split] = {"n": n, "sin_dim": bad}
    return np.array(W), np.array(H), np.array(split_of), counts


def stats(a, nd=1):
    r = lambda x: round(float(x), nd)
    p = lambda q: r(np.percentile(a, q))
    return {"min": r(a.min()), "p1": p(1), "p5": p(5), "q1": p(25),
            "median": p(50), "mean": r(a.mean()), "q3": p(75),
            "p95": p(95), "p99": p(99), "max": r(a.max()), "std": r(a.std())}


def figure(W, H, area, aspect, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#3a4550",
        "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#e6ebf0",
        "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 110,
    })
    DET = "#0e7c8b"; REC = "#b5670a"; INK = "#1a2129"; MED = "#c81e2c"
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    panels = [
        (ax[0, 0], W, "Ancho de placa (px)", DET, "(a)"),
        (ax[0, 1], H, "Alto de placa (px)", DET, "(b)"),
        (ax[1, 0], area, "Área de placa (px²)", REC, "(c)"),
        (ax[1, 1], aspect, "Relación de aspecto (ancho / alto)", REC, "(d)"),
    ]
    for a, data, xlabel, color, tag in panels:
        lo, hi = np.percentile(data, 0.5), np.percentile(data, 99.5)
        a.hist(data, bins=60, range=(lo, hi), color=color, alpha=0.82, edgecolor="white", linewidth=0.25)
        med = np.median(data); q1, q3 = np.percentile(data, [25, 75])
        a.axvline(med, color=MED, lw=1.6, label="mediana %.0f" % med if data is not aspect else "mediana %.2f" % med)
        a.axvspan(q1, q3, color=MED, alpha=0.07)
        a.set_xlabel(xlabel); a.set_ylabel("nº de placas")
        a.yaxis.set_major_locator(MaxNLocator(5))
        a.yaxis.set_major_formatter(lambda v, _: ("%.0fk" % (v/1000)) if v >= 1000 else "%.0f" % v)
        a.text(0.015, 0.94, tag, transform=a.transAxes, fontsize=11, fontweight="bold", color=INK, va="top")
        a.legend(loc="upper right", frameon=False, fontsize=9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Distribución de tamaño de placa — %s crops" % f"{len(W):,}".replace(",", " "),
                 fontsize=13.5, fontweight="bold", x=0.5, y=1.035, color=INK)
    fig.text(0.5, 0.995, "Dataset ALPR peruano · tamaño derivado de la anotación silver del detector",
             ha="center", fontsize=9.5, color="#5a6572")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../Peruvian_ALPR_Dataset_final")
    ap.add_argument("--out", default="fig_plate_size_distribution")
    a = ap.parse_args()
    W, H, split_of, counts = collect(a.dataset)
    area = W * H
    aspect = W / np.maximum(H, 1)
    res = {
        "n_total": int(len(W)), "counts_por_split": counts,
        "ancho_px": stats(W), "alto_px": stats(H), "area_px2": stats(area),
        "aspect": stats(aspect, nd=3),
    }
    json.dump(res, open("full_size_distribution.json", "w"), ensure_ascii=False, indent=2)
    figure(W, H, area, aspect, a.out)
    print("N crops procesados:", res["n_total"])
    for split, c in counts.items():
        print("  %-5s n=%d  sin_dim=%d" % (split, c["n"], c["sin_dim"]))
    for k in ("ancho_px", "alto_px", "area_px2", "aspect"):
        s = res[k]
        print("%-9s min=%s q1=%s median=%s q3=%s p99=%s max=%s" %
              (k, s["min"], s["q1"], s["median"], s["q3"], s["p99"], s["max"]))
    print("figura ->", a.out + ".png / .pdf")


if __name__ == "__main__":
    main()
