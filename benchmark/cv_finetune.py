#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validacion cruzada del ajuste fino: estima la VARIANZA del procedimiento, no de una corrida.

POR QUE HACE FALTA. Todas las cifras anteriores proceden de un unico entrenamiento. Una sola
corrida no permite distinguir una diferencia real entre arquitecturas de la variabilidad
propia del entrenamiento (inicializacion, orden de los lotes, particion concreta del gold).
Sin esa estimacion no se puede afirmar que una arquitectura sea mejor: solo que lo fue una vez.

DISEÑO. Validacion cruzada de k pliegues sobre el gold de ajuste, factorial completo:

    Factor A (arquitectura): LPRNet, CRNN          -> 2 niveles
    Factor B (pliegue):      1..k                   -> k bloques
    Respuesta:               exact-match sobre Gold-B

Cada pliegue entrena con (k-1)/k del gold y evalua sobre el MISMO conjunto externo, de modo
que la unica fuente de variacion entre celdas es el procedimiento, no el conjunto de prueba.
El pliegue actua como **bloque**: es un diseño de medidas repetidas, y el contraste adecuado
es un ANOVA de dos vias sin replicacion (o su equivalente no parametrico, Friedman).

Uso:
    python cv_finetune.py --k 5 --epochs 15
"""
import argparse, csv, json, random, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bench_utils import NUM_CLASSES, ctc_greedy_decode, metrics, norm_plate
from datasets import PlateDataset, collate_ctc
from models import build_model


def greedy(logits):
    idx = logits.argmax(2).permute(1, 0)
    return [ctc_greedy_decode(r.tolist()) for r in idx]


@torch.no_grad()
def evaluar(model, loader, dev):
    model.eval()
    preds, truths = [], []
    for imgs, _, _, plates in loader:
        for p, t in zip(greedy(model(imgs.to(dev)).cpu()), plates):
            preds.append(p)
            truths.append(t)
    return metrics(preds, truths)


def entrenar(ckpt, tsv_train, loaders, epochs, lr, batch, dev):
    ck = torch.load(ckpt, map_location=dev)
    model = build_model(ck["model"], NUM_CLASSES).to(dev)
    model.load_state_dict(ck["state_dict"])
    H, W = ck["input_size"]
    tl = DataLoader(PlateDataset(tsv_train, H, W, augment=True), batch, shuffle=True,
                    num_workers=2, collate_fn=collate_ctc, drop_last=True)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for imgs, tgt, largos, _ in tl:
            imgs, tgt = imgs.to(dev), tgt.to(dev)
            logp = model(imgs).log_softmax(2)
            T, B, _ = logp.shape
            loss = crit(logp, tgt, torch.full((B,), T, dtype=torch.long), largos)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return {nombre: evaluar(model, dl, dev) for nombre, (dl, _) in loaders.items()}, (H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/rec_labels_gold_all.tsv")
    ap.add_argument("--externo", default="data/rec_labels_gold2.tsv")
    ap.add_argument("--retenido", default="data/rec_labels_gold2_clean.tsv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="../docs/paper_materials/17_validacion_cruzada.json")
    a = ap.parse_args()

    filas = [r for r in csv.reader(open(a.gold, encoding="utf-8"), delimiter="\t") if len(r) >= 2]
    # Los pliegues se forman por MATRICULA, no por recorte: dos recortes de la misma placa
    # en pliegues distintos serian fuga dentro de la propia validacion cruzada.
    por_placa = {}
    for r in filas:
        por_placa.setdefault(norm_plate(r[1]), []).append(r)
    placas = sorted(por_placa)
    rng = random.Random(a.seed)
    rng.shuffle(placas)
    pliegues = [placas[i::a.k] for i in range(a.k)]
    print(f"Gold de ajuste: {len(filas)} recortes de {len(placas)} matriculas")
    print(f"Validacion cruzada de {a.k} pliegues, agrupados por matricula "
          f"({', '.join(str(len(p)) for p in pliegues)} matriculas por pliegue)\n")

    arquitecturas = [("LPRNet", "ckpt/lprnet.pt"), ("CRNN", "ckpt/crnn.pt")]
    tmp = Path(tempfile.mkdtemp())
    resultados = {n: {"gold_B": [], "retenido": [], "interno": []} for n, _ in arquitecturas}

    for nombre, ckpt in arquitecturas:
        if not Path(ckpt).exists():
            print(f"  falta {ckpt}, se omite {nombre}")
            continue
        ck = torch.load(ckpt, map_location="cpu")
        H, W = ck["input_size"]
        loaders = {
            "gold_B": (DataLoader(PlateDataset(a.externo, H, W), 256, shuffle=False,
                                  num_workers=2, collate_fn=collate_ctc), a.externo),
            "retenido": (DataLoader(PlateDataset(a.retenido, H, W), 256, shuffle=False,
                                    num_workers=2, collate_fn=collate_ctc), a.retenido),
        }
        for i in range(a.k):
            test_placas = set(pliegues[i])
            tr_rows = [r for p, rs in por_placa.items() if p not in test_placas for r in rs]
            te_rows = [r for p in test_placas for r in por_placa[p]]
            f_tr = tmp / f"{nombre}_tr{i}.tsv"
            f_te = tmp / f"{nombre}_te{i}.tsv"
            for f, rows in ((f_tr, tr_rows), (f_te, te_rows)):
                with open(f, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)
            loaders["interno"] = (DataLoader(PlateDataset(str(f_te), H, W), 256, shuffle=False,
                                             num_workers=2, collate_fn=collate_ctc), str(f_te))
            m, _ = entrenar(ckpt, str(f_tr), loaders, a.epochs, a.lr, a.batch, a.device)
            for clave in ("gold_B", "retenido", "interno"):
                resultados[nombre][clave].append(m[clave]["exact_match"])
            print(f"  {nombre:7s} pliegue {i+1}/{a.k}: gold_B={m['gold_B']['exact_match']:.4f}  "
                  f"retenido={m['retenido']['exact_match']:.4f}  "
                  f"interno={m['interno']['exact_match']:.4f}", flush=True)

    # --- Resumen y contrastes
    print(f"\n{'='*68}\nRESUMEN (exact-match sobre Gold-B)")
    print(f"  {'Arquitectura':14s} {'media':>8s} {'sd':>8s} {'min':>8s} {'max':>8s} {'IC 95% media':>20s}")
    resumen = {}
    from scipy import stats as sps
    for nombre in resultados:
        v = np.array(resultados[nombre]["gold_B"])
        if v.size == 0:
            continue
        sd = float(v.std(ddof=1))
        ic = sps.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sd / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)
        resumen[nombre] = {"media": float(v.mean()), "sd": sd, "min": float(v.min()),
                           "max": float(v.max()), "ic95_media": [float(ic[0]), float(ic[1])],
                           "valores": [float(x) for x in v]}
        print(f"  {nombre:14s} {v.mean():8.4f} {sd:8.4f} {v.min():8.4f} {v.max():8.4f}"
              f"   [{ic[0]:.4f}, {ic[1]:.4f}]")

    contrastes = {}
    if len(resumen) == 2:
        n1, n2 = list(resumen)
        v1 = np.array(resultados[n1]["gold_B"])
        v2 = np.array(resultados[n2]["gold_B"])
        # Los pliegues son los mismos para ambas arquitecturas -> muestras APAREADAS.
        t = sps.ttest_rel(v1, v2)
        w = sps.wilcoxon(v1, v2) if len(v1) >= 6 else None
        d = float((v1 - v2).mean() / (v1 - v2).std(ddof=1)) if (v1 - v2).std(ddof=1) > 0 else float("inf")
        contrastes = {"comparacion": f"{n1} vs {n2}",
                      "diferencia_media": float((v1 - v2).mean()),
                      "t_pareada": {"t": float(t.statistic), "gl": len(v1) - 1, "p": float(t.pvalue)},
                      "d_cohen_pareada": d}
        if w is not None:
            contrastes["wilcoxon"] = {"W": float(w.statistic), "p": float(w.pvalue)}
        print(f"\nCONTRASTE ENTRE ARQUITECTURAS (pliegues como bloques, muestras apareadas)")
        print(f"  {n1} - {n2} = {(v1-v2).mean():+.4f}")
        print(f"  t pareada: t({len(v1)-1}) = {t.statistic:.3f}  p = {t.pvalue:.4f}")
        print(f"  d de Cohen pareada = {d:.2f}")
        if w is not None:
            print(f"  Wilcoxon de rangos con signo: W = {w.statistic:.1f}  p = {w.pvalue:.4f}")

    reporte = {"k": a.k, "epocas": a.epochs, "lr": a.lr, "semilla": a.seed,
               "agrupacion": "pliegues formados por matricula, no por recorte",
               "conjunto_externo": a.externo, "conjunto_retenido": a.retenido,
               "resultados": resultados, "resumen_gold_B": resumen,
               "contraste_arquitecturas": contrastes}
    dest = Path(a.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {dest}")


if __name__ == "__main__":
    main()
