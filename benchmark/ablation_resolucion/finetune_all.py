#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ajuste fino de LPRNet y CRNN a 32x100, sobre las tres semillas de la base silver.

Llama a ../finetune.py para cada checkpoint base y despues evalua el resultado
sobre Gold-B, que es la cifra principal del articulo y que finetune.py no mide
(su --test es el conjunto retenido).

Reporta media y desviacion tipica entre semillas, que es lo unico interpretable
dado que la variabilidad de entrenamiento a esta escala ronda el punto.

Uso (desde benchmark/):
    python ablation_resolucion/finetune_all.py
    python ablation_resolucion/finetune_all.py --modelos lprnet --semillas 0
"""
import os
import sys
import json
import time
import argparse
import subprocess

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from datasets import PlateDataset, collate_ctc                    # noqa: E402
from models import build_model                                    # noqa: E402
from bench_utils import NUM_CLASSES, ctc_greedy_decode, metrics   # noqa: E402


def hhmm(s):
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


@torch.no_grad()
def evaluar(ckpt, tsv, device, batch=256):
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
    m = metrics(preds, truths)
    m["len_err"] = sum(1 for p in preds if len(p) != 6) / max(1, len(preds))
    return m, preds, truths


def media_sd(v):
    if not v:
        return 0.0, 0.0
    mu = sum(v) / len(v)
    if len(v) < 2:
        return mu, 0.0
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=os.path.join(HERE, "resultados", "ckpt"),
                    help="carpeta con lprnet_32x100_s*.pt y crnn_32x100_s*.pt")
    ap.add_argument("--modelos", default="lprnet,crnn")
    ap.add_argument("--semillas", default="0,1,2")
    ap.add_argument("--res", default="32x100")
    ap.add_argument("--gold", default=os.path.join(BENCH, "data", "rec_labels_gold_all.tsv"),
                    help="datos de ajuste, reciben gradiente")
    ap.add_argument("--eval", default=os.path.join(BENCH, "data", "rec_labels_gold.tsv"),
                    help="Gold-A, seleccion de epoca, no se reporta")
    ap.add_argument("--retenido", default=os.path.join(BENCH, "data", "rec_labels_gold2_clean.tsv"),
                    help="control de fuga")
    ap.add_argument("--goldb", default=os.path.join(BENCH, "data", "rec_labels_gold2.tsv"),
                    help="Gold-B, evaluacion principal")
    ap.add_argument("--out", default=os.path.join(HERE, "resultados_ft"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--evaluar-solo", action="store_true", dest="evaluar_solo",
                    help="no reentrena: reevalua los *_ft.pt que ya existen")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    modelos = [m.strip() for m in a.modelos.split(",") if m.strip()]
    semillas = [int(s) for s in a.semillas.split(",") if s.strip()]
    dest = os.path.join(a.out, "finetune_%s.json" % a.res)

    # Se conservan las filas ya calculadas de otras corridas: si se relanza con
    # --modelos crnn no deben perderse las de lprnet.
    filas = []
    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            filas = json.load(f)
        print("recuperadas %d filas previas de %s" % (len(filas), dest), flush=True)

    def guardar(nueva):
        clave = (nueva["modelo"], nueva["res"], nueva["semilla"])
        for j, v in enumerate(filas):
            if (v["modelo"], v["res"], v["semilla"]) == clave:
                filas[j] = nueva
                break
        else:
            filas.append(nueva)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(filas, f, ensure_ascii=False, indent=1)

    total, i, t0 = len(modelos) * len(semillas), 0, time.time()
    for name in modelos:
        for s in semillas:
            i += 1
            base = os.path.join(a.bases, "%s_%s_s%d.pt" % (name, a.res, s))
            if not os.path.exists(base):
                print("  falta %s, se omite" % base, flush=True)
                continue
            salida = os.path.join(a.out, "ckpt", "%s_%s_s%d_ft.pt" % (name, a.res, s))
            rep = os.path.join(a.out, "%s_%s_s%d_ft.json" % (name, a.res, s))

            print("\n" + "=" * 78, flush=True)
            print("[%d/%d]  %s  semilla %d  |  transcurrido %s"
                  % (i, total, name.upper(), s, hhmm(time.time() - t0)), flush=True)
            print("=" * 78, flush=True)

            if a.evaluar_solo:
                if not os.path.exists(salida):
                    print("  falta %s, se omite" % salida, flush=True)
                    continue
                print("  solo evaluacion, no se reentrena", flush=True)
            else:
                cmd = [sys.executable, os.path.join(BENCH, "finetune.py"),
                       "--ckpt", base, "--gold", a.gold, "--eval", a.eval,
                       "--test", a.retenido, "--epochs", str(a.epochs),
                       "--lr", str(a.lr), "--batch", str(a.batch),
                       "--out", salida, "--reporte", rep, "--device", a.device]
                r = subprocess.run(cmd, cwd=BENCH)
                if r.returncode != 0:
                    print("  finetune.py fallo (codigo %d), se omite" % r.returncode, flush=True)
                    continue

            antes, _, _ = evaluar(base, a.goldb, a.device)
            despues, pred, truth = evaluar(salida, a.goldb, a.device)
            ret, _, _ = evaluar(salida, a.retenido, a.device)
            print("  GOLD-B  base %.4f  ->  afinado %.4f  (%+.2f pp)   CER %.4f -> %.4f   len!=6 %.4f"
                  % (antes["exact_match"], despues["exact_match"],
                     100 * (despues["exact_match"] - antes["exact_match"]),
                     antes["cer"], despues["cer"], despues["len_err"]), flush=True)

            guardar({
                "modelo": name, "res": a.res, "semilla": s,
                "base_goldb": round(antes["exact_match"], 5),
                "ft_goldb": round(despues["exact_match"], 5),
                "delta_pp": round(100 * (despues["exact_match"] - antes["exact_match"]), 3),
                "ft_cer": round(despues["cer"], 5),
                "ft_len_err": round(despues["len_err"], 5),
                "ft_retenido": round(ret["exact_match"], 5),
                "ft_retenido_cer": round(ret["cer"], 5),
                "ckpt": salida, "pred": pred, "truth": truth,
            })

    print("\n" + "=" * 78, flush=True)
    print("RESUMEN  (Gold-B, n = %d)   completado en %s"
          % (len(filas[0]["truth"]) if filas else 0, hhmm(time.time() - t0)), flush=True)
    print("=" * 78, flush=True)
    print("%-8s %-22s %-22s %-10s %-10s"
          % ("modelo", "base", "afinado", "delta", "retenido"), flush=True)
    for name in modelos:
        rs = [r for r in filas if r["modelo"] == name]
        if not rs:
            continue
        mb, sb = media_sd([r["base_goldb"] for r in rs])
        mf, sf = media_sd([r["ft_goldb"] for r in rs])
        mr, sr = media_sd([r["ft_retenido"] for r in rs])
        print("%-8s %.4f +/- %.4f      %.4f +/- %.4f      %+6.2f pp   %.4f +/- %.4f"
              % (name, mb, sb, mf, sf, 100 * (mf - mb), mr, sr), flush=True)
    print("\nresultados en %s" % dest, flush=True)


if __name__ == "__main__":
    main()
