#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contraste estadistico de las comparaciones del articulo, respetando el apareamiento.

Todos los sistemas de reconocimiento se evaluan sobre EL MISMO gold de 782 placas, y todos
los detectores sobre EL MISMO test. Son por tanto **muestras apareadas**: cada placa aporta
un par de resultados correlacionados (si una placa es intrinsecamente dificil, lo es para
los dos modelos). Tratarlas como independientes -- un test de dos proporciones, o comparar
intervalos de confianza por separado -- ignora esa correlacion y pierde potencia: puede
declarar "no significativo" un efecto real.

El contraste correcto para exact-match apareado es la **prueba de McNemar**, que solo mira
los casos discordantes (los que un modelo acierta y el otro falla). Aqui se usa la version
exacta (binomial), valida con pocos discordantes, donde la aproximacion chi-cuadrado falla.

Los intervalos de confianza de las DIFERENCIAS se calculan por bootstrap apareado: se
remuestrean placas (no predicciones), preservando el apareamiento.

Uso:
    python paired_stats.py --out ../docs/paper_materials/08_estadistica.json
"""
import argparse, csv, json, math, os, sys
from pathlib import Path

import numpy as np
import torch

from bench_utils import norm_plate
from corrector import correct as corregir
from models import build_model

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
I2C = {i + 1: c for i, c in enumerate(CHARS)}


def ctc_greedy(probs):
    """(T, C) -> texto, colapsando repetidos y quitando el blank (indice 0)."""
    out, prev = [], -1
    for t in range(probs.shape[0]):
        idx = int(probs[t].argmax())
        if idx != 0 and idx != prev:
            out.append(I2C.get(idx, ""))
        prev = idx
    return "".join(out)


def cargar_gold(tsv):
    """[(ruta, verdad)] en orden estable."""
    filas = []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    filas.append((row[0], p))
    return filas


def predecir_ckpt(ckpt_path, rutas, device="cpu"):
    """Predicciones de un checkpoint sobre las rutas dadas, en el mismo orden."""
    import cv2
    ck = torch.load(ckpt_path, map_location=device)
    H, W = ck["input_size"]
    modelo = build_model(ck["model"], ck["num_classes"]).to(device)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()

    preds = []
    B = 64
    for i in range(0, len(rutas), B):
        lote = []
        for r in rutas[i:i + B]:
            img = cv2.imread(r)
            if img is None:
                img = np.zeros((H, W, 3), np.uint8)
            img = cv2.resize(img, (W, H)).astype(np.float32) / 127.5 - 1.0
            lote.append(img.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(lote)).to(device)
        with torch.no_grad():
            logits = modelo(x)                      # (T, B, C)
        probs = logits.permute(1, 0, 2).cpu().numpy()
        preds.extend(ctc_greedy(p) for p in probs)
    return preds


def cargar_paddle(tsv, rutas):
    """Predicciones de PaddleOCR (cruda y corregida) alineadas por nombre de archivo."""
    por_nombre = {}
    with open(tsv, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            por_nombre[row["nombre_archivo"]] = row
    crudas, corregidas, verdades = [], [], []
    faltan = 0
    for r in rutas:
        fila = por_nombre.get(os.path.basename(r))
        if fila is None:
            faltan += 1
            crudas.append(""); corregidas.append(""); verdades.append(None)
            continue
        crudas.append(norm_plate(fila["placa_ocr"]))
        corregidas.append(norm_plate(fila["placa_corregida"]))
        verdades.append(norm_plate(fila["placa_correcta"]))
    return crudas, corregidas, verdades, faltan


def mcnemar_exacto(a, b):
    """Prueba de McNemar exacta entre dos vectores booleanos apareados.

    b01 = A falla y B acierta ; b10 = A acierta y B falla. Bajo H0 los discordantes se
    reparten como una binomial(n_disc, 0.5). Se devuelve el p-valor bilateral exacto.
    """
    b10 = int(np.sum(a & ~b))
    b01 = int(np.sum(~a & b))
    n = b10 + b01
    if n == 0:
        return {"b10": 0, "b01": 0, "n_discordantes": 0, "p": 1.0}
    k = min(b10, b01)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b10": b10, "b01": b01, "n_discordantes": n, "p": min(1.0, 2 * cola)}


def bootstrap_diferencia(a, b, n_iter=10000, seed=42):
    """IC 95 % bootstrap APAREADO de la diferencia de exactitud (a - b).

    Se remuestrean placas, no predicciones: cada replica toma los resultados de ambos
    modelos sobre las mismas placas, conservando la correlacion.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_iter, n))
    difs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def wilson(k, n, z=1.96):
    """IC 95 % de Wilson para una proporcion (mejor que el normal con n moderado)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / d
    margen = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/rec_labels_gold.tsv")
    ap.add_argument("--paddle", default="../dataset_nuevo_validacion/resultados_verificacion.tsv")
    ap.add_argument("--out", default="../docs/paper_materials/08_estadistica.json")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    filas = cargar_gold(args.gold)
    rutas = [r for r, _ in filas]
    verdad = [v for _, v in filas]
    print(f"Gold: {len(filas)} placas (mismo conjunto para todos los sistemas)")

    # --- Verificacion del apareamiento: la verdad-terreno debe coincidir entre fuentes.
    crudas, corregidas, verdad_paddle, faltan = cargar_paddle(args.paddle, rutas)
    if faltan:
        sys.exit(f"{faltan} placas del gold no estan en {args.paddle}: el apareamiento se rompe")
    discrepancias = sum(1 for a, b in zip(verdad, verdad_paddle) if a != b)
    if discrepancias:
        sys.exit(f"La verdad-terreno difiere en {discrepancias} placas entre las dos fuentes")
    print("Apareamiento verificado: misma verdad-terreno en ambas fuentes, mismo orden\n")

    sistemas = {
        "PaddleOCR (crudo)": crudas,
        "PaddleOCR + corrector": corregidas,
    }
    for nombre, ckpt in (("CRNN", "ckpt/crnn.pt"),
                         ("LPRNet", "ckpt/lprnet.pt"),
                         ("LPRNet afinado", "ckpt/lprnet_ft_2lotes.pt")):
        if Path(ckpt).exists():
            print(f"  infiriendo {nombre} ...")
            sistemas[nombre] = predecir_ckpt(ckpt, rutas)
    if "LPRNet afinado" in sistemas:
        sistemas["LPRNet afinado + corrector"] = [
            corregir(p) if p else p for p in sistemas["LPRNet afinado"]
        ]

    aciertos = {n: np.array([p == v for p, v in zip(preds, verdad)])
                for n, preds in sistemas.items()}

    n = len(verdad)
    print(f"\n{'Sistema':30s} {'Aciertos':>10s} {'Exactitud':>10s} {'IC 95 % (Wilson)':>22s}")
    resumen = {}
    for nombre, ok in aciertos.items():
        k = int(ok.sum())
        lo, hi = wilson(k, n)
        resumen[nombre] = {"aciertos": k, "n": n, "exactitud": k / n,
                           "ic95_wilson": [lo, hi]}
        print(f"{nombre:30s} {k:6d}/{n:<4d} {k/n:9.4f}   [{lo:.4f}, {hi:.4f}]")

    # --- Comparaciones apareadas de interes para el articulo
    pares = [
        ("LPRNet afinado", "PaddleOCR + corrector",
         "El estudiante afinado supera al maestro que genero sus etiquetas"),
        ("LPRNet afinado", "LPRNet",
         "Efecto del ajuste fino con gold por seleccion activa"),
        ("PaddleOCR + corrector", "PaddleOCR (crudo)",
         "Aporte del corrector determinista sobre el OCR generico"),
        ("LPRNet", "CRNN",
         "Comparacion entre arquitecturas compactas destiladas"),
        ("LPRNet afinado + corrector", "LPRNet afinado",
         "El corrector ya no aporta tras el ajuste fino"),
    ]

    print(f"\n{'Comparacion (A vs B)':52s} {'dif':>8s} {'IC 95 %':>20s} {'b10/b01':>9s} {'p':>10s}")
    comparaciones = []
    for a_nom, b_nom, hipotesis in pares:
        if a_nom not in aciertos or b_nom not in aciertos:
            continue
        a, b = aciertos[a_nom], aciertos[b_nom]
        mc = mcnemar_exacto(a, b)
        lo, hi = bootstrap_diferencia(a, b, n_iter=args.boot)
        dif = float(a.mean() - b.mean())
        etiqueta = f"{a_nom} vs {b_nom}"
        print(f"{etiqueta:52s} {dif:+8.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{mc['b10']:4d}/{mc['b01']:<4d} {mc['p']:10.2e}")
        comparaciones.append({
            "A": a_nom, "B": b_nom, "hipotesis": hipotesis,
            "diferencia": dif, "ic95_bootstrap_apareado": [lo, hi],
            "mcnemar": mc,
            "significativo_0.05": mc["p"] < 0.05,
        })

    reporte = {
        "n_placas": n,
        "diseno": "muestras apareadas (todos los sistemas sobre el mismo gold)",
        "pruebas": {
            "exactitud": "IC de Wilson por sistema",
            "comparacion": "McNemar exacta (binomial) sobre discordantes",
            "diferencia": f"bootstrap apareado, {args.boot} replicas, remuestreo de placas",
        },
        "sistemas": resumen,
        "comparaciones": comparaciones,
    }
    destino = Path(args.out).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {destino}")


if __name__ == "__main__":
    main()
