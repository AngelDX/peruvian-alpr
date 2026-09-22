#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcula DE UNA VEZ todas las cifras que citan los manuscritos y los dossiers.

Existe para evitar el problema que motivó su creación: cada documento citaba una medición
hecha en un momento distinto, con un checkpoint distinto o sobre un subconjunto distinto, y
las cifras dejaron de cuadrar entre sí. Aquí se fija el checkpoint canónico y se recalcula
todo sobre los tres conjuntos, con los contrastes apareados incluidos.

Salida: JSON con la tabla maestra. Cualquier cifra de los documentos debe poder rastrearse
hasta este archivo.

Uso:
    python cifras_canonicas.py
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

import numpy as np

from bench_utils import norm_plate, metrics
from corrector import correct as corregir
from paired_stats import predecir_ckpt, mcnemar_exacto, bootstrap_diferencia, wilson

CROP_RE = re.compile(r".+_\(\s*\d+,\s*\d+,\s*3\)_(.+)\.jpg$", re.IGNORECASE)

CKPT_CANONICO = "ckpt/lprnet_ft_v2.pt"

CONJUNTOS = [
    ("retenido", "data/rec_labels_gold2_clean.tsv",
     "689 recortes sin solape con ajuste ni con seleccion de epoca"),
    ("gold_B", "data/rec_labels_gold2.tsv",
     "1130 recortes, una placa por matricula, distribucion del corpus"),
    ("gold_A", "data/rec_labels_gold.tsv",
     "782 recortes muestreados por frecuencia de paso (vista por evento)"),
]


def cargar(tsv):
    filas = [r for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t") if len(r) >= 2]
    return [r[0] for r in filas], [norm_plate(r[1]) for r in filas]


def crudo(fn):
    m = CROP_RE.match(Path(fn).name)
    return norm_plate(m.group(1)) if m else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=CKPT_CANONICO)
    ap.add_argument("--out", default="../docs/paper_materials/15_cifras_canonicas.json")
    a = ap.parse_args()

    reporte = {"checkpoint_canonico": a.ckpt, "conjuntos": {}}
    print(f"Checkpoint canonico: {a.ckpt}\n")

    for nombre, tsv, desc in CONJUNTOS:
        if not Path(tsv).exists():
            print(f"  falta {tsv}, se omite {nombre}")
            continue
        rutas, verdad = cargar(tsv)
        n = len(rutas)

        sistemas = {}
        cr = [crudo(p) for p in rutas]
        # PaddleOCR crudo solo es recuperable del nombre en los conjuntos gold2.
        if all(cr):
            sistemas["PaddleOCR (crudo)"] = cr
            sistemas["PaddleOCR + corrector"] = [
                norm_plate(corregir(c[:3] + "-" + c[3:])) if len(c) == 6 else "" for c in cr]
        for et, ck in (("CRNN", "ckpt/crnn.pt"),
                       ("LPRNet base", "ckpt/lprnet.pt"),
                       ("LPRNet afinado", a.ckpt)):
            if Path(ck).exists():
                sistemas[et] = predecir_ckpt(ck, rutas)

        print(f"=== {nombre} (n = {n}) — {desc}")
        datos = {}
        aciertos = {}
        for et, preds in sistemas.items():
            pn = [norm_plate(p) for p in preds]
            ok = np.array([p == v for p, v in zip(pn, verdad)])
            aciertos[et] = ok
            m = metrics(pn, verdad)
            lo, hi = wilson(int(ok.sum()), n)
            datos[et] = {"aciertos": int(ok.sum()), "n": n,
                         "exact_match": m["exact_match"], "cer": m["cer"],
                         "ic95_wilson": [lo, hi]}
            print(f"  {et:24s} {ok.sum():5d}/{n:<5d} {m['exact_match']:.4f}  "
                  f"CER {m['cer']:.4f}  IC[{lo:.4f},{hi:.4f}]")

        comparaciones = []
        pares = [("LPRNet afinado", "PaddleOCR + corrector"),
                 ("LPRNet afinado", "LPRNet base"),
                 ("PaddleOCR + corrector", "PaddleOCR (crudo)"),
                 ("LPRNet base", "CRNN")]
        for x, y in pares:
            if x not in aciertos or y not in aciertos:
                continue
            mc = mcnemar_exacto(aciertos[x], aciertos[y])
            lo, hi = bootstrap_diferencia(aciertos[x], aciertos[y])
            d = float(aciertos[x].mean() - aciertos[y].mean())
            comparaciones.append({"A": x, "B": y, "diferencia": d,
                                  "ic95": [lo, hi], "mcnemar": mc})
            print(f"    {x} vs {y}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"b10={mc['b10']} b01={mc['b01']} p={mc['p']:.2e}")

        # Estratificacion luminica, solo donde hay metadata
        estratos = {}
        meta_path = Path("data/rec_labels_gold2_meta.csv")
        if meta_path.exists():
            meta = {r["path"]: r for r in csv.DictReader(open(meta_path, encoding="utf-8"))}
            luz = np.array([meta.get(p, {}).get("day_night", "") for p in rutas])
            for c in ("dia", "crepusculo", "noche"):
                m_ = luz == c
                if m_.sum() >= 10:
                    estratos.setdefault("luz", {})[c] = {
                        et: {"aciertos": int(ok[m_].sum()), "n": int(m_.sum()),
                             "exact_match": float(ok[m_].mean())}
                        for et, ok in aciertos.items()}
            if "luz" in estratos:
                print("    luz: " + "  ".join(
                    f"{c}(n={estratos['luz'][c]['LPRNet afinado']['n']})="
                    f"{estratos['luz'][c]['LPRNet afinado']['exact_match']:.3f}"
                    for c in estratos["luz"]))

        reporte["conjuntos"][nombre] = {"tsv": tsv, "descripcion": desc, "n": n,
                                        "sistemas": datos, "comparaciones": comparaciones,
                                        "estratos": estratos}
        print()

    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tabla maestra: {dest}")


if __name__ == "__main__":
    main()
