#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analisis inferencial de los reconocedores: omnibus, post-hoc y modelo factorial.

CONTRASTE PRINCIPAL
    H0: los k modelos tienen el mismo desempeño.
    H1: al menos uno difiere del resto.

La prueba adecuada NO es un ANOVA. La respuesta es **binaria** (cada placa se acierta o no) y
**las mediciones estan apareadas**: los k modelos se evaluan sobre las mismas placas. Para ese
diseño —k muestras relacionadas con respuesta dicotomica— el contraste omnibus es la
**Q de Cochran**, que generaliza McNemar a mas de dos modelos. Un ANOVA clasico supondria
respuesta continua, homocedasticidad y observaciones independientes; ninguna se cumple aqui.

POST-HOC
    Rechazar H0 solo dice que alguno difiere. Para saber cuales, se hacen las k(k-1)/2
    comparaciones por pares con **McNemar exacta** y se corrige el error de tipo I por
    **Holm-Bonferroni**, uniformemente mas potente que Bonferroni y valido sin supuestos de
    independencia entre las comparaciones.

TAMAÑO DEL EFECTO
    El p-valor no mide magnitud. Se reporta la **razon de discordantes** (odds ratio de
    McNemar, b10/b01) y la **g de Cohen** para datos apareados, ademas de la diferencia
    absoluta con su intervalo por bootstrap apareado.

DISEÑO FACTORIAL
    Se ajusta un modelo lineal generalizado binomial con enlace logit sobre el diseño
    completo modelo x clase luminica x frecuencia de la placa, con errores robustos
    agrupados por placa (cada matricula aporta k observaciones correlacionadas). Permite
    contrastar efectos principales e interacciones, que es lo que un ANOVA factorial haria
    si la respuesta fuese continua.

Uso:
    python analisis_inferencial.py
"""
import argparse, csv, itertools, json, re
from pathlib import Path

import numpy as np
from scipy import stats

from bench_utils import norm_plate
from corrector import correct as corregir
from paired_stats import predecir_ckpt, mcnemar_exacto, bootstrap_diferencia, wilson

CROP_RE = re.compile(r".+_\(\s*\d+,\s*\d+,\s*3\)_(.+)\.jpg$", re.IGNORECASE)


def crudo_de_nombre(fn):
    m = CROP_RE.match(Path(fn).name)
    return norm_plate(m.group(1)) if m else ""


def cochran_q(matriz):
    """Q de Cochran para k muestras relacionadas con respuesta binaria.

    matriz: (n sujetos, k tratamientos) de 0/1.
    Bajo H0 (todos los tratamientos tienen la misma proporcion de exito), Q ~ chi2(k-1).
    Solo aportan informacion las filas no constantes: las placas que todos aciertan o todos
    fallan se cancelan, igual que los concordantes en McNemar.
    """
    x = np.asarray(matriz, dtype=float)
    n, k = x.shape
    col = x.sum(axis=0)          # aciertos por modelo
    fil = x.sum(axis=1)          # aciertos por placa
    num = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2)
    den = k * fil.sum() - (fil ** 2).sum()
    if den == 0:
        return {"Q": 0.0, "gl": k - 1, "p": 1.0, "filas_informativas": 0}
    Q = num / den
    gl = k - 1
    p = float(stats.chi2.sf(Q, gl))
    informativas = int(((fil > 0) & (fil < k)).sum())
    return {"Q": float(Q), "gl": gl, "p": p, "filas_informativas": informativas}


def holm(pvals):
    """Correccion de Holm-Bonferroni. Devuelve los p ajustados en el orden original."""
    m = len(pvals)
    orden = np.argsort(pvals)
    ajust = np.empty(m)
    acumulado = 0.0
    for i, idx in enumerate(orden):
        v = (m - i) * pvals[idx]
        acumulado = max(acumulado, v)      # monotonia
        ajust[idx] = min(1.0, acumulado)
    return ajust


def g_cohen(b10, b01):
    """g de Cohen para datos apareados: desviacion de 0.5 en el reparto de discordantes."""
    n = b10 + b01
    return abs(b10 / n - 0.5) if n else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/rec_labels_gold2_meta.csv")
    ap.add_argument("--ckpt-ft", default="ckpt/lprnet_ft_v2.pt")
    ap.add_argument("--out", default="../docs/paper_materials/16_analisis_inferencial.json")
    ap.add_argument("--alfa", type=float, default=0.05)
    ap.add_argument("--matriz", default="results/matriz_gold2.csv",
                    help="CSV con el acierto por placa y sistema (vacio para no volcarlo)")
    a = ap.parse_args()

    filas = list(csv.DictReader(open(a.meta, encoding="utf-8")))
    rutas = [r["path"] for r in filas]
    verdad = [norm_plate(r["placa"]) for r in filas]
    n = len(filas)

    # --- Sistemas
    sistemas = {}
    cr = [crudo_de_nombre(r["fn"]) for r in filas]
    sistemas["PaddleOCR crudo"] = cr
    sistemas["PaddleOCR+corr"] = [norm_plate(corregir(c[:3] + "-" + c[3:])) if len(c) == 6 else ""
                                  for c in cr]
    for et, ck in (("CRNN", "ckpt/crnn.pt"), ("LPRNet", "ckpt/lprnet.pt"),
                   ("LPRNet-ft", a.ckpt_ft)):
        if Path(ck).exists():
            print(f"  infiriendo {et} ...")
            sistemas[et] = predecir_ckpt(ck, rutas)

    nombres = list(sistemas)
    X = np.column_stack([[norm_plate(p) == v for p, v in zip(sistemas[m], verdad)]
                         for m in nombres]).astype(int)
    k = X.shape[1]

    # Matriz de aciertos por placa. Se vuelca porque los contrastes pareados y el
    # bootstrap la necesitan placa a placa, y regenerarla exige reinferir.
    if a.matriz:
        Path(a.matriz).parent.mkdir(parents=True, exist_ok=True)
        with open(a.matriz, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["placa", "verdad"] + nombres + ["pred_" + m for m in nombres])
            for i, fila in enumerate(filas):
                w.writerow([fila.get("fn", i), verdad[i]]
                           + [int(X[i, j]) for j in range(k)]
                           + [norm_plate(sistemas[m][i]) for m in nombres])
        print(f"  matriz por placa -> {a.matriz}")

    print(f"\nDiseño: {n} placas x {k} modelos, respuesta binaria, medidas repetidas\n")
    print(f"{'Modelo':18s} {'Aciertos':>12s} {'Proporcion':>11s} {'IC 95% Wilson':>20s}")
    descr = {}
    for j, m in enumerate(nombres):
        ac = int(X[:, j].sum())
        lo, hi = wilson(ac, n)
        descr[m] = {"aciertos": ac, "n": n, "p": ac / n, "ic95": [lo, hi]}
        print(f"{m:18s} {ac:6d}/{n:<5d} {ac/n:11.4f}   [{lo:.4f}, {hi:.4f}]")

    # --- OMNIBUS
    q = cochran_q(X)
    print(f"\n{'='*66}\nOMNIBUS — Q de Cochran")
    print(f"  H0: los {k} modelos tienen la misma proporcion de acierto")
    print(f"  Q = {q['Q']:.2f}   gl = {q['gl']}   p = {q['p']:.3e}")
    print(f"  placas informativas (no constantes): {q['filas_informativas']} de {n}")
    print(f"  -> {'SE RECHAZA H0' if q['p'] < a.alfa else 'no se rechaza H0'} al {a.alfa:.0%}")

    # --- POST-HOC
    pares = list(itertools.combinations(range(k), 2))
    crudos, detalle = [], []
    for i, j in pares:
        mc = mcnemar_exacto(X[:, i].astype(bool), X[:, j].astype(bool))
        lo, hi = bootstrap_diferencia(X[:, i].astype(bool), X[:, j].astype(bool))
        crudos.append(mc["p"])
        detalle.append({"A": nombres[i], "B": nombres[j],
                        "dif": float(X[:, i].mean() - X[:, j].mean()),
                        "ic95": [lo, hi], "b10": mc["b10"], "b01": mc["b01"],
                        "p_crudo": mc["p"],
                        "razon_discordantes": (mc["b10"] / mc["b01"]) if mc["b01"] else float("inf"),
                        "g_cohen": g_cohen(mc["b10"], mc["b01"])})
    ajust = holm(np.array(crudos))
    for d, pa in zip(detalle, ajust):
        d["p_holm"] = float(pa)
        d["significativo"] = bool(pa < a.alfa)

    print(f"\nPOST-HOC — McNemar exacta con correccion de Holm-Bonferroni ({len(pares)} pares)")
    print(f"  {'A vs B':34s} {'dif':>8s} {'b10/b01':>10s} {'p crudo':>10s} {'p Holm':>10s} {'g':>6s}")
    for d in sorted(detalle, key=lambda x: x["p_holm"]):
        marca = " *" if d["significativo"] else "  "
        print(f"  {d['A'] + ' vs ' + d['B']:34s} {d['dif']:+8.4f} "
              f"{d['b10']:4d}/{d['b01']:<5d} {d['p_crudo']:10.2e} {d['p_holm']:10.2e} "
              f"{d['g_cohen']:6.3f}{marca}")
    print(f"  (*) significativo al {a.alfa:.0%} tras correccion")

    # --- ORDENACION: ranking medio, base de la lectura grafica
    rangos = np.zeros(k)
    for f in range(n):
        fila = X[f]
        # rango medio por fila (1 = mejor). Empates reciben el rango promedio.
        rangos += stats.rankdata(-fila)
    rangos /= n
    orden = np.argsort(rangos)
    print(f"\nRANGO MEDIO (1 = mejor; base de la comparacion grafica)")
    for idx in orden:
        print(f"  {nombres[idx]:18s} {rangos[idx]:.3f}")

    # --- DISEÑO FACTORIAL: modelo x luz x frecuencia
    factorial = None
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        def bucket(v):
            v = int(v or 0)
            return "f1_2" if v <= 2 else "f3_9" if v < 10 else "f10_49" if v < 50 else "f50"

        largo = []
        for f_idx, r in enumerate(filas):
            for j, m in enumerate(nombres):
                largo.append({"acierto": int(X[f_idx, j]), "modelo": m,
                              "luz": r["luz"] or "dia", "frec": bucket(r["frames_de_la_placa"]),
                              "placa": r["placa"]})
        df = pd.DataFrame(largo)
        df["modelo"] = pd.Categorical(df["modelo"], categories=nombres)

        # GLM binomial con errores agrupados por placa: cada matricula aporta k
        # observaciones correlacionadas, y ignorarlo subestimaria los errores estandar.
        mod = smf.glm("acierto ~ C(modelo) * C(luz) + C(modelo) * C(frec)",
                      data=df, family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial())
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["placa"]})

        # Contrastes tipo Wald de cada termino
        terminos = {}
        for pref, etq in (("C(modelo)[", "modelo"), ("C(luz)[", "luz"), ("C(frec)[", "frecuencia")):
            idx = [i for i, nm in enumerate(res.params.index)
                   if nm.startswith(pref) and ":" not in nm]
            if idx:
                R = np.zeros((len(idx), len(res.params)))
                for fila_i, c in enumerate(idx):
                    R[fila_i, c] = 1
                w = res.wald_test(R, scalar=True)
                terminos[etq] = {"chi2": float(w.statistic), "gl": int(len(idx)),
                                 "p": float(w.pvalue)}
        for pref, etq in (("C(modelo)[", "modelo x luz"), ("C(modelo)[", "modelo x frecuencia")):
            clave = ":C(luz)" if "luz" in etq else ":C(frec)"
            idx = [i for i, nm in enumerate(res.params.index) if clave in nm]
            if idx:
                R = np.zeros((len(idx), len(res.params)))
                for fila_i, c in enumerate(idx):
                    R[fila_i, c] = 1
                w = res.wald_test(R, scalar=True)
                terminos[etq] = {"chi2": float(w.statistic), "gl": int(len(idx)),
                                 "p": float(w.pvalue)}

        print(f"\nDISEÑO FACTORIAL — GLM binomial (logit), errores agrupados por matricula")
        print(f"  {'Termino':22s} {'chi2':>10s} {'gl':>4s} {'p':>12s}")
        for etq, t in terminos.items():
            print(f"  {etq:22s} {t['chi2']:10.2f} {t['gl']:4d} {t['p']:12.3e}")
        factorial = {"terminos": terminos, "n_obs": len(df),
                     "formula": "acierto ~ C(modelo)*C(luz) + C(modelo)*C(frec)",
                     "covarianza": "cluster por matricula"}
    except Exception as e:
        print(f"\n(modelo factorial no ajustado: {e})")

    reporte = {
        "n_placas": n, "k_modelos": k, "modelos": nombres,
        "descriptivos": descr,
        "omnibus_cochran_q": q,
        "posthoc": detalle,
        "correccion": "Holm-Bonferroni",
        "rango_medio": {nombres[i]: float(rangos[i]) for i in range(k)},
        "factorial": factorial,
        "nota_metodologica": (
            "Respuesta binaria y medidas repetidas sobre las mismas placas: el omnibus es la Q "
            "de Cochran, no un ANOVA. El post-hoc son McNemar exactas con correccion de Holm."),
    }
    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {dest}")


if __name__ == "__main__":
    main()
