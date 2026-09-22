#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenera todas las cifras de la Seccion 4.3 con los modelos a 32x100 y tres semillas.

El manuscrito tiene hoy una incoherencia: la Metodologia declara 32x100 y tres
semillas, pero los Resultados siguen siendo de 24x94 y corrida unica. Este script
reevalua los cinco sistemas sobre Gold-A, Gold-B y el retenido usando los
checkpoints nuevos, y vuelca las tablas en LaTeX listas para sustituir.

Los dos sistemas PaddleOCR no dependen de la resolucion: sus lecturas salen del
nombre de archivo, como en analisis_inferencial.py, y no se reinfieren.

Uso (desde benchmark/):
    python regenerar_cifras_32x100.py --smoke     # solo semilla 0, para comprobar
    python regenerar_cifras_32x100.py
"""
import os
import re
import csv
import sys
import json
import argparse
import itertools
from math import comb

import numpy as np
import torch
from torch.utils.data import DataLoader

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

from datasets import PlateDataset, collate_ctc                     # noqa: E402
from models import build_model                                     # noqa: E402
from bench_utils import NUM_CLASSES, ctc_greedy_decode, metrics, norm_plate  # noqa: E402
from corrector import correct as corregir                       # noqa: E402

ABL = os.path.join(BENCH, "ablation_resolucion")
BASES = os.path.join(ABL, "resultados", "ckpt")          # *_32x100_s*.pt
FT = os.path.join(ABL, "resultados_ft", "ckpt")          # *_32x100_s*_ft.pt

CONJ = {
    "gold_A": os.path.join(BENCH, "data", "rec_labels_gold.tsv"),
    "gold_B": os.path.join(BENCH, "data", "rec_labels_gold2.tsv"),
    "retenido": os.path.join(BENCH, "data", "rec_labels_gold2_clean.tsv"),
}
META_B = os.path.join(BENCH, "data", "rec_labels_gold2_meta.csv")


# Gold-A es un caso aparte: sus recortes NO llevan en el nombre la salida del OCR.
# En 174 de sus 782 filas el nombre y la lectura real difieren (el nombre viene de
# otra pasada de deteccion), asi que el crudo hay que leerlo del fichero de
# verificacion. En Gold-B y en el retenido el nombre si es la lectura.
VERIF_A = os.path.join(os.path.dirname(BENCH), "dataset_nuevo_validacion",
                       "resultados_verificacion.tsv")


def _tabla_verificacion():
    if not os.path.exists(VERIF_A):
        return {}
    with open(VERIF_A, encoding="utf-8") as f:
        return {r["nombre_archivo"].strip('"'): r["salida_paddle_cruda"]
                for r in csv.DictReader(f, delimiter="\t")}


_VERIF = _tabla_verificacion()


def crudo_de_nombre(fn):
    """Lectura del OCR crudo para un recorte, por fichero de verificacion o nombre."""
    if fn in _VERIF:
        return norm_plate(_VERIF[fn])
    m = re.search(r"_([A-Z0-9-]{6,8})\.jpg$", fn)
    return norm_plate(m.group(1)) if m else ""


@torch.no_grad()
def predecir(ckpt, tsv, device, batch=256):
    ck = torch.load(ckpt, map_location=device)
    model = build_model(ck["model"], NUM_CLASSES).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    H, W = ck["input_size"]
    dl = DataLoader(PlateDataset(tsv, H, W), batch, shuffle=False,
                    num_workers=2, collate_fn=collate_ctc)
    preds, truths = [], []
    for imgs, _, _, plates in dl:
        idx = model(imgs.to(device)).argmax(2).permute(1, 0).cpu()
        preds.extend(ctc_greedy_decode(r.tolist()) for r in idx)
        truths.extend(plates)
    return preds, truths


def paddle(tsv):
    """Lecturas de PaddleOCR crudo y corregido, desde el nombre de archivo."""
    crudo, corr, verdad = [], [], []
    with open(tsv, encoding="utf-8") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            if not linea:
                continue
            ruta, placa = linea.split("\t")
            c = crudo_de_nombre(re.split(r"[\\/]", ruta)[-1])
            crudo.append(c)
            corr.append(norm_plate(corregir(c[:3] + "-" + c[3:])) if len(c) == 6 else "")
            verdad.append(norm_plate(placa))
    return crudo, corr, verdad


def media_sd(v):
    if not v:
        return 0.0, 0.0
    mu = sum(v) / len(v)
    if len(v) < 2:
        return mu, 0.0
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - r) / d, (c + r) / d


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


def cochran_q(X):
    n, k = X.shape
    fil, col = X.sum(1), X.sum(0)
    N = X.sum()
    num = (k - 1) * (k * (col ** 2).sum() - N ** 2)
    den = k * N - (fil ** 2).sum()
    Q = num / den if den else 0.0
    inf = int(((fil > 0) & (fil < k)).sum())
    return Q, k - 1, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", default="0,1,2")
    ap.add_argument("--res", default="32x100")
    ap.add_argument("--out", default=os.path.join(BENCH, "results", "cifras_32x100.json"))
    ap.add_argument("--tex", default=os.path.join(BENCH, "tables", "latex", "tablas_32x100.tex"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    semillas = [0] if a.smoke else [int(x) for x in a.semillas.split(",")]

    print("dispositivo %s | semillas %s | resolución %s" % (a.device, semillas, a.res), flush=True)

    # ---- sistemas neuronales, por conjunto y semilla
    modelos = {
        "CRNN": (BASES, "crnn_%s_s%d.pt"),
        "LPRNet": (BASES, "lprnet_%s_s%d.pt"),
        "LPRNet afinado": (FT, "lprnet_%s_s%d_ft.pt"),
        "CRNN afinado": (FT, "crnn_%s_s%d_ft.pt"),
    }
    salida = {}
    for nombre_conj, tsv in CONJ.items():
        print("\n== %s ==" % nombre_conj, flush=True)
        crudo, corr, verdad = paddle(tsv)
        n = len(verdad)
        aciertos = {
            "PaddleOCR crudo": [np.array([p == t for p, t in zip(crudo, verdad)], int)],
            "PaddleOCR + corrector": [np.array([p == t for p, t in zip(corr, verdad)], int)],
        }
        cer = {
            "PaddleOCR crudo": [metrics(crudo, verdad)["cer"]],
            "PaddleOCR + corrector": [metrics(corr, verdad)["cer"]],
        }
        for et, (carpeta, patron) in modelos.items():
            aciertos[et], cer[et] = [], []
            for s in semillas:
                ck = os.path.join(carpeta, patron % (a.res, s))
                if not os.path.exists(ck):
                    print("   falta %s" % os.path.basename(ck), flush=True)
                    continue
                pr, tr = predecir(ck, tsv, a.device)
                aciertos[et].append(np.array([p == t for p, t in zip(pr, tr)], int))
                cer[et].append(metrics(pr, tr)["cer"])
                print("   %-16s s%d  exacta %.4f" % (et, s, aciertos[et][-1].mean()), flush=True)

        fila = {}
        for et in aciertos:
            ex = [v.mean() for v in aciertos[et]]
            mu, sd = media_sd(ex)
            lo, hi = wilson(int(round(mu * n)), n)
            mc, _ = media_sd(cer[et])
            fila[et] = {"n": n, "exacta": mu, "sd": sd, "ic95": [lo, hi],
                        "cer": mc, "corridas": len(ex)}
            print("   %-22s %.4f +/- %.4f   CER %.4f" % (et, mu, sd, mc), flush=True)
        salida[nombre_conj] = fila
        if nombre_conj in ("gold_A", "gold_B"):
            salida["_aciertos_" + nombre_conj] = aciertos

        # contrastes solo sobre Gold-B y con la semilla 0, que es la reportable
        if nombre_conj == "gold_B":
            nombres = list(aciertos)
            X = np.column_stack([aciertos[e][0] for e in nombres])
            Q, gl, inf = cochran_q(X)
            print("\n   Q de Cochran = %.2f  gl = %d  informativas = %d" % (Q, gl, inf), flush=True)
            pares, ps, det = list(itertools.combinations(range(len(nombres)), 2)), [], []
            for i, j in pares:
                p, b10, b01 = mcnemar(X[:, i], X[:, j])
                ps.append(p)
                det.append((nombres[i], nombres[j], 100 * (X[:, i].mean() - X[:, j].mean()), b10, b01))
            aj = holm(ps)
            salida["omnibus"] = {"Q": Q, "gl": gl, "informativas": inf}
            # matriz de aciertos por placa, que consumen las figuras
            mz = os.path.join(BENCH, "results", "matriz_gold2_32x100.csv")
            os.makedirs(os.path.dirname(mz), exist_ok=True)
            with open(mz, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["placa"] + nombres)
                for i in range(X.shape[0]):
                    w.writerow([verdad[i]] + [int(X[i, j]) for j in range(X.shape[1])])
            print("   matriz por placa -> %s" % os.path.basename(mz), flush=True)

            salida["posthoc"] = [
                {"A": A, "B": B, "delta_pp": d, "b10": b10, "b01": b01, "p_holm": q}
                for (A, B, d, b10, b01), q in zip(det, aj)]

    # ================================================================
    # Desglose por subgrupos sobre Gold-B, usando los metadatos
    # ================================================================
    if os.path.exists(META_B) and "gold_B" in salida:
        print("\n== desglose por subgrupos (Gold-B) ==", flush=True)
        with open(META_B, encoding="utf-8") as f:
            meta = {r["fn"]: r for r in csv.DictReader(f)}

        # orden de filas del TSV de Gold-B, para alinear con las predicciones
        fns = []
        with open(CONJ["gold_B"], encoding="utf-8") as f:
            for linea in f:
                linea = linea.rstrip("\n")
                if linea:
                    fns.append(re.split(r"[\\/]", linea.split("\t")[0])[-1])

        def bucket_frec(v):
            v = int(float(v))
            return "1--2" if v <= 2 else "3--9" if v <= 9 else "10--49" if v <= 49 else "50+"

        LUZ = {"dia": "Día", "crepusculo": "Crepúsculo", "noche": "Noche"}
        grupos = {
            "frecuencia": [bucket_frec(meta[f]["frames_de_la_placa"]) if f in meta else "?"
                           for f in fns],
            "luz": [LUZ.get(meta[f]["luz"], "?") if f in meta else "?" for f in fns],
        }
        # El area NO se estratifica aqui. Gold-B sobremuestrea la noche a proposito
        # (24 % frente al 3.2 % del corpus) y el area esta correlacionada con la luz:
        # el cuartil grande tiene el triple de capturas nocturnas que el pequeno,
        # probablemente porque la placa retrorreflectante florece bajo iluminacion
        # infrarroja y el detector dibuja una caja mayor. Estratificar por area en
        # Gold-B mide luz. El analisis de tamano va sobre Gold-A, mas abajo.
        areas_B = np.array([float(meta[f]["area_px2"]) if f in meta else np.nan for f in fns])
        qs_B = np.nanpercentile(areas_B, [25, 50, 75])
        qB = np.array(["Q1" if x <= qs_B[0] else "Q2" if x <= qs_B[1] else
                       "Q3" if x <= qs_B[2] else "Q4" for x in areas_B])
        noche = np.array([meta[f]["luz"] == "noche" if f in meta else False for f in fns])
        salida["confusor_area_luz_gold_B"] = {
            q: round(float(noche[qB == q].mean()), 4) for q in ("Q1", "Q2", "Q3", "Q4")}
        print("   proporción nocturna por cuartil de área en Gold-B: %s"
              % salida["confusor_area_luz_gold_B"], flush=True)

        ORDEN_G = {"frecuencia": ["1--2", "3--9", "10--49", "50+"],
                   "luz": ["Día", "Crepúsculo", "Noche"]}
        TITULO = {"frecuencia": "número de pasadas del vehículo",
                  "luz": "clase lumínica, por elevación solar real"}

        desglose, bloques = {}, []
        aciertos_B = salida["_aciertos_gold_B"]
        for g, etiquetas in grupos.items():
            niveles = ORDEN_G[g]
            idx = {lv: np.array([e == lv for e in etiquetas]) for lv in niveles}
            desglose[g] = {"n": {lv: int(idx[lv].sum()) for lv in niveles}, "sistemas": {}}
            L = [r"\begin{table}[H]",
                 r"\caption{Coincidencia exacta sobre \emph{Gold}-B por %s. Media de tres semillas"
                 % TITULO[g],
                 r"en los modelos entrenados.\label{tab:%s}}" % ({"frecuencia": "frecuencia", "luz": "luz"}[g]),
                 r"\begin{tabularx}{\textwidth}{X" + "r" * len(niveles) + "}", r"\toprule",
                 r"\textbf{Sistema} & " + " & ".join(
                     r"\textbf{%s} (n = %d)" % (lv, idx[lv].sum()) for lv in niveles) + r" \\",
                 r"\midrule"]
            for et in ["LPRNet afinado", "CRNN afinado", "LPRNet", "CRNN",
                       "PaddleOCR + corrector", "PaddleOCR crudo"]:
                if et not in aciertos_B:
                    continue
                vals = []
                for lv in niveles:
                    v = [arr[idx[lv]].mean() for arr in aciertos_B[et] if idx[lv].any()]
                    vals.append(sum(v) / len(v) if v else float("nan"))
                desglose[g]["sistemas"][et] = dict(zip(niveles, vals))
                L.append("%s & %s \\\\" % (et, " & ".join("%.1f\\,\\%%" % (100 * v) for v in vals)))
            L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
            bloques.append("\n".join(L))
            print("   %s: %s" % (g, {lv: int(idx[lv].sum()) for lv in niveles}), flush=True)
        salida["desglose"] = desglose

        with open(a.tex.replace(".tex", "_subgrupos.tex"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(bloques) + "\n")
        print("   subgrupos -> %s" % a.tex.replace(".tex", "_subgrupos.tex"))

        # ---- tamano de matricula, sobre Gold-A
        # El area sale de las dimensiones codificadas en el nombre del recorte,
        # (alto, ancho, 3), identicas al area_px2 del metadato en Gold-B.
        dim = re.compile(r"_\((\d+), (\d+), 3\)_")
        fa, ar = [], []
        with open(CONJ["gold_A"], encoding="utf-8") as f:
            for linea in f:
                linea = linea.rstrip("\n")
                if not linea:
                    continue
                nom = re.split(r"[\\/]", linea.split("\t")[0])[-1]
                m = dim.search(nom)
                fa.append(nom)
                ar.append(int(m.group(1)) * int(m.group(2)) if m else np.nan)
        ar = np.array(ar, dtype=float)
        aciertos_A = salida.get("_aciertos_gold_A", {})
        if aciertos_A and np.isfinite(ar).sum() > 0.9 * len(ar):
            qa = np.nanpercentile(ar, [25, 50, 75])
            qA = np.array(["Q1" if x <= qa[0] else "Q2" if x <= qa[1] else
                           "Q3" if x <= qa[2] else "Q4" for x in ar])
            niv = ["Q1", "Q2", "Q3", "Q4"]
            idxA = {lv: (qA == lv) for lv in niv}
            L = [r"\begin{table}[H]",
                 r"\caption{Coincidencia exacta sobre \emph{Gold}-A por cuartil de área de la",
                 r"matrícula. Se usa \emph{Gold}-A y no \emph{Gold}-B porque este último",
                 r"sobremuestrea la noche y el área queda confundida con la condición lumínica.",
                 r"Media de tres semillas en los modelos entrenados.\label{tab:area}}",
                 r"\begin{tabularx}{\textwidth}{Xrrrr}", r"\toprule",
                 r"\textbf{Sistema} & " + " & ".join(
                     r"\textbf{%s} ($\le$%d\,px$^2$, n = %d)" % (lv, int(q), idxA[lv].sum())
                     for lv, q in zip(niv, list(qa) + [int(np.nanmax(ar))])) + r" \\",
                 r"\midrule"]
            tabla_area = {}
            for et in ["LPRNet afinado", "CRNN afinado", "LPRNet", "CRNN",
                       "PaddleOCR + corrector", "PaddleOCR crudo"]:
                if et not in aciertos_A:
                    continue
                vals = [float(np.mean([arr[idxA[lv]].mean() for arr in aciertos_A[et]]))
                        for lv in niv]
                tabla_area[et] = dict(zip(niv, vals))
                L.append("%s & %s \\\\" % (et, " & ".join("%.1f\\,\\%%" % (100 * v) for v in vals)))
            L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
            bloques.append("\n".join(L))
            salida.setdefault("desglose", {})["area_gold_A"] = {
                "n": {lv: int(idxA[lv].sum()) for lv in niv},
                "cortes_px2": [int(x) for x in qa], "sistemas": tabla_area}
            print("   área (Gold-A): cortes %s" % [int(x) for x in qa], flush=True)

        # ---- modelo factorial binomial con errores agrupados por matrícula.
        # GEE con estructura de independencia equivale a un GLM binomial con
        # errores robustos por conglomerado, que es lo que declara la metodología.
        try:
            import pandas as pd
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            filas = []
            for et, arrs in aciertos_B.items():
                y = arrs[0]                       # semilla de referencia
                for i, yi in enumerate(y):
                    filas.append({"y": int(yi), "sistema": et, "luz": grupos["luz"][i],
                                  "frec": grupos["frecuencia"][i], "placa": i})
            df = pd.DataFrame(filas)
            modelo = smf.gee("y ~ C(sistema)*C(frec) + C(sistema)*C(luz)", "placa", data=df,
                             family=sm.families.Binomial(),
                             cov_struct=sm.cov_struct.Independence()).fit()
            # statsmodels >= 0.14 devuelve arrays en las celdas de la tabla, no
            # escalares, de ahi el aplanado antes de convertir.
            def esc(x):
                v = np.asarray(x).ravel()
                return float(v[0]) if v.size else float("nan")

            # scalar=False conserva el comportamiento actual y silencia el FutureWarning
            tabla = modelo.wald_test_terms(skip_single=False, scalar=False).table
            salida["factorial"] = {
                str(i): {"chi2": esc(r["statistic"]), "gl": int(esc(r["df_constraint"])),
                         "p": esc(r["pvalue"])}
                for i, r in tabla.iterrows()}
            print("\n   modelo factorial (Wald por término):", flush=True)
            for k, v in salida["factorial"].items():
                print("      %-24s chi2 %8.2f  gl %2d  p %.3g"
                      % (k, v["chi2"], v["gl"], v["p"]), flush=True)
        except Exception as e:
            print("\n   modelo factorial omitido: %s" % e, flush=True)

    for k in ("_aciertos_gold_A", "_aciertos_gold_B"):
        salida.pop(k, None)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=float)
    print("\nJSON -> %s" % a.out)

    # ---- tabla principal en LaTeX
    os.makedirs(os.path.dirname(a.tex), exist_ok=True)
    orden = ["LPRNet afinado", "CRNN afinado", "LPRNet", "CRNN",
             "PaddleOCR + corrector", "PaddleOCR crudo"]
    L = [r"% Generado por regenerar_cifras_32x100.py. Sustituye a tab:reconocimiento.",
         r"\begin{table}[H]",
         r"\caption{Coincidencia exacta y CER sobre las dos verdades-terreno, a $32\times100$\,px.",
         r"Media y desviación típica de tres semillas para los modelos entrenados.\label{tab:reconocimiento}}",
         r"\begin{tabularx}{\textwidth}{Xrrrr}", r"\toprule",
         r"& \multicolumn{2}{c}{\textbf{\emph{Gold}-B}} & \multicolumn{2}{c}{\textbf{\emph{Gold}-A}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"\textbf{Sistema} & \textbf{Exacta} & \textbf{CER} & \textbf{Exacta} & \textbf{CER} \\",
         r"\midrule"]
    for et in orden:
        b, A = salida.get("gold_B", {}).get(et), salida.get("gold_A", {}).get(et)
        if not b:
            continue
        def fmt(d):
            if not d:
                return "--"
            return ("%.2f\\,\\%%" % (100 * d["exacta"])) + (
                " $\\pm$ %.2f" % (100 * d["sd"]) if d["corridas"] > 1 else "")
        L.append("%s & %s & %.4f & %s & %.4f \\\\" %
                 (et, fmt(b), b["cer"], fmt(A), A["cer"] if A else 0.0))
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    with open(a.tex, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("LaTeX -> %s" % a.tex)
    print("\nListo. Tablas principales en %s y desgloses en %s."
          % (os.path.basename(a.tex), os.path.basename(a.tex.replace(".tex", "_subgrupos.tex"))))


if __name__ == "__main__":
    main()
