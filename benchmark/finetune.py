#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning de un reconocedor pre-entrenado con gold, con separacion estricta entre
seleccion de modelo y evaluacion final.

POR QUE ESTA SEPARACION. La version anterior guardaba el punto de control que maximizaba el
acierto sobre el MISMO conjunto que luego se reportaba como resultado. Eso es seleccion de
modelo sobre el conjunto de prueba: la cifra publicada era «la mejor de N epocas» y no una
estimacion independiente, de modo que estaba optimisticamente sesgada.

Ahora intervienen tres conjuntos disjuntos en su proposito:

    --gold    datos de ajuste          -> gradiente
    --eval    conjunto de SELECCION    -> decide que epoca se guarda (nunca se reporta)
    --test    conjunto RETENIDO        -> se mide una sola vez, con el checkpoint ya elegido

El conjunto de --test no interviene en ninguna decision: ni en el gradiente, ni en la parada,
ni en la eleccion de epoca. Su cifra es la que debe ir al articulo.

Se avisa si hay solape de matriculas entre conjuntos: al ser etiquetas ruidosas, agrupar por
cadena no garantiza separar vehiculos, y conviene saber cuanto solape queda.

Uso:
    python finetune.py --ckpt ckpt/lprnet.pt --gold data/rec_labels_gold_all.tsv \
        --eval data/rec_labels_gold.tsv --test data/rec_labels_gold2_clean.tsv \
        --epochs 25 --lr 3e-4 --out ckpt/lprnet_ft_v2.pt
"""
import argparse, csv, json
from pathlib import Path

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
def evaluate(model, loader, dev):
    model.eval()
    preds, truths = [], []
    for imgs, _, _, plates in loader:
        for p, t in zip(greedy(model(imgs.to(dev)).cpu()), plates):
            preds.append(p)
            truths.append(t)
    return metrics(preds, truths)


def placas(tsv):
    s = set()
    if not Path(tsv).exists():
        return s
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    s.add(p)
    return s


def avisar_solape(gold, ev, test):
    """El solape no bloquea, pero debe quedar registrado en el log del entrenamiento."""
    g, e, t = placas(gold), placas(ev), placas(test)
    print("Matriculas unicas -> ajuste=%d  seleccion=%d  retenido=%d" % (len(g), len(e), len(t)))
    for nombre, a, b in (("ajuste vs seleccion", g, e),
                         ("ajuste vs RETENIDO", g, t),
                         ("seleccion vs RETENIDO", e, t)):
        n = len(a & b)
        pct = (100 * n / len(b)) if b else 0
        marca = "  <-- contamina la cifra final" if ("RETENIDO" in nombre and n) else ""
        print("  %-22s %4d  (%.1f %% del segundo)%s" % (nombre, n, pct, marca))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gold", required=True, help="datos de ajuste (reciben gradiente)")
    ap.add_argument("--eval", default="data/rec_labels_gold.tsv",
                    help="conjunto de SELECCION de epoca; su cifra no se reporta")
    ap.add_argument("--test", default="data/rec_labels_gold2_clean.tsv",
                    help="conjunto RETENIDO; se mide una sola vez al final")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reporte", default=None, help="JSON con el historial y las cifras finales")
    a = ap.parse_args()

    avisar_solape(a.gold, a.eval, a.test)
    print()

    ck = torch.load(a.ckpt, map_location=a.device)
    model = build_model(ck["model"], NUM_CLASSES).to(a.device)
    model.load_state_dict(ck["state_dict"])
    H, W = ck["input_size"]

    tl = DataLoader(PlateDataset(a.gold, H, W, augment=True), a.batch, shuffle=True,
                    num_workers=2, collate_fn=collate_ctc, drop_last=True)
    ev = DataLoader(PlateDataset(a.eval, H, W), 256, shuffle=False,
                    num_workers=2, collate_fn=collate_ctc)
    te = DataLoader(PlateDataset(a.test, H, W), 256, shuffle=False,
                    num_workers=2, collate_fn=collate_ctc)

    base_ev = evaluate(model, ev, a.device)
    base_te = evaluate(model, te, a.device)
    print("BASE (%s)  seleccion: exact=%.4f  |  RETENIDO: exact=%.4f CER=%.4f"
          % (ck["model"], base_ev["exact_match"], base_te["exact_match"], base_te["cer"]), flush=True)

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    mejor = base_ev["exact_match"]
    mejor_ep = 0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    historial = []

    for ep in range(1, a.epochs + 1):
        model.train()
        for imgs, tgt, largos, _ in tl:
            imgs = imgs.to(a.device)
            tgt = tgt.to(a.device)
            logp = model(imgs).log_softmax(2)
            T, B, _ = logp.shape
            loss = crit(logp, tgt, torch.full((B,), T, dtype=torch.long), largos)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # SOLO el conjunto de seleccion decide. El retenido no se toca dentro del bucle.
        m = evaluate(model, ev, a.device)
        historial.append({"epoca": ep, "seleccion_exact": m["exact_match"], "seleccion_cer": m["cer"]})
        marca = ""
        if m["exact_match"] > mejor:
            mejor = m["exact_match"]
            mejor_ep = ep
            mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marca = " *"
        print("ep %2d/%d  seleccion_exact=%.4f  CER=%.4f%s"
              % (ep, a.epochs, m["exact_match"], m["cer"], marca), flush=True)

    # --- Evaluacion final: una sola vez, con la epoca ya decidida.
    model.load_state_dict(mejor_estado)
    final_te = evaluate(model, te, a.device)
    final_ev = evaluate(model, ev, a.device)

    torch.save({"model": ck["model"], "state_dict": mejor_estado, "input_size": ck["input_size"],
                "num_classes": NUM_CLASSES,
                "epoca_elegida": mejor_ep,
                "seleccion_exact": final_ev["exact_match"],
                "test_exact": final_te["exact_match"],
                "conjunto_seleccion": a.eval,
                "conjunto_retenido": a.test}, a.out)

    print("\n" + "=" * 68)
    print("Epoca elegida: %d (por el conjunto de SELECCION, exact=%.4f)" % (mejor_ep, final_ev["exact_match"]))
    print("%-34s %10s %10s" % ("", "base", "afinado"))
    print("%-34s %10.4f %10.4f" % ("seleccion (no reportable)", base_ev["exact_match"], final_ev["exact_match"]))
    print("%-34s %10.4f %10.4f   <- cifra del articulo"
          % ("RETENIDO (%d recortes)" % len(te.dataset), base_te["exact_match"], final_te["exact_match"]))
    print("%-34s %10.4f %10.4f" % ("RETENIDO · CER", base_te["cer"], final_te["cer"]))
    print("Checkpoint: %s" % a.out)

    if a.reporte:
        Path(a.reporte).parent.mkdir(parents=True, exist_ok=True)
        Path(a.reporte).write_text(json.dumps({
            "ckpt_base": a.ckpt, "salida": a.out,
            "conjunto_ajuste": a.gold, "conjunto_seleccion": a.eval, "conjunto_retenido": a.test,
            "epocas": a.epochs, "lr": a.lr, "batch": a.batch,
            "epoca_elegida": mejor_ep,
            "base": {"seleccion": base_ev, "retenido": base_te},
            "afinado": {"seleccion": final_ev, "retenido": final_te},
            "historial": historial,
            "nota": ("el conjunto retenido no interviene en el gradiente ni en la eleccion de "
                     "epoca; su cifra es la estimacion independiente"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Reporte: %s" % a.reporte)


if __name__ == "__main__":
    main()
