#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablacion de la resolucion de normalizacion de la ROI.

Entrena cada combinacion (modelo x resolucion) sobre las etiquetas silver,
elige la epoca por validacion y evalua sobre Gold-B. Guarda un JSON con las
predicciones por matricula para que report_ablation.py pueda hacer los
contrastes pareados de McNemar.

Uso tipico (desde benchmark/):
    python ablation_resolucion/run_ablation.py --epochs 30

Prueba rapida antes de lanzar la corrida larga:
    python ablation_resolucion/run_ablation.py --smoke
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)

from datasets import PlateDataset, collate_ctc              # noqa: E402
from bench_utils import NUM_CLASSES, ctc_greedy_decode, metrics  # noqa: E402
from models_res import build_model_res, seq_len             # noqa: E402

# (alto, ancho, etiqueta). El aspecto nativo de la placa peruana es 1.98.
RESOLUCIONES = [
    (24, 94, "linea base LPRNet"),
    (32, 100, "linea base CRNN"),
    (48, 96, "nativa 2:1"),
    (64, 128, "2:1 ampliada"),
    (80, 160, "2:1 muy ampliada"),
    (96, 320, "referencia coreana 3.33:1"),
]


def greedy(logits):
    idx = logits.argmax(2).permute(1, 0)
    return [ctc_greedy_decode(r.tolist()) for r in idx]


class Log(object):
    """Escribe a consola y a un fichero por configuracion, sin buffering."""

    def __init__(self, ruta):
        self.f = open(ruta, "w", encoding="utf-8", buffering=1)

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")

    def close(self):
        self.f.close()


def diagnostico(preds, truths):
    """Metricas de la epoca mas el diagnostico especifico de esta ablacion.

    len_err es la tasa de lecturas cuya longitud no es 6. Es la prediccion
    concreta de la hipotesis: si el aplastamiento vertical hace que la CTC
    codiciosa fusione caracteres adyacentes, esta tasa debe bajar al respetar
    el aspecto nativo 2:1.
    """
    m = metrics(preds, truths)
    n = max(1, len(preds))
    m["len_err"] = sum(1 for p in preds if len(p) != 6) / n
    m["cortas"] = sum(1 for p in preds if len(p) < 6)
    m["largas"] = sum(1 for p in preds if len(p) > 6)
    return m


def hhmm(seg):
    seg = int(seg)
    return "%d:%02d:%02d" % (seg // 3600, (seg % 3600) // 60, seg % 60)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, truths = [], []
    for imgs, _, _, plates in loader:
        out = model(imgs.to(device)).cpu()
        preds.extend(greedy(out))
        truths.extend(plates)
    return preds, truths


@torch.no_grad()
def latency_ms(model, h, w, device, n=100):
    model.eval()
    x = torch.zeros(1, 3, h, w, device=device)
    for _ in range(10):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def train_one(name, h, w, args, device, etiqueta=""):
    torch.manual_seed(args.seed)
    model = build_model_res(name, NUM_CLASSES, (h, w)).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    logs = os.path.join(args.out, "logs")
    os.makedirs(logs, exist_ok=True)
    log = Log(os.path.join(logs, "%s_%dx%d_s%d.log" % (name, h, w, args.seed)))

    tr = PlateDataset(os.path.join(args.data, "rec_labels_train.tsv"), h, w, augment=True)
    va = PlateDataset(os.path.join(args.data, "rec_labels_val.tsv"), h, w)
    if args.limit:
        tr.items = tr.items[: args.limit]
        va.items = va.items[: max(200, args.limit // 10)]
    gold = PlateDataset(args.gold, h, w)

    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_ctc, drop_last=True)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_ctc)
    gl = DataLoader(gold, args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_ctc)

    log("%s  %dx%d  aspecto %.2f  (%s)" % (name.upper(), h, w, w / h, etiqueta))
    log("  T=%d  params=%.3fM  train=%d  val=%d  gold=%d  batch=%d  lr=%g  epocas=%d  semilla=%d"
        % (seq_len(name, w), n_par / 1e6, len(tr), len(va), len(gold),
           args.batch, args.lr, args.epochs, args.seed))
    log("  %-5s %-8s %-9s %-8s %-9s %-8s %-8s %s"
        % ("ep", "loss", "val_exact", "val_cer", "len!=6", "lr", "seg", "ETA"))

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val, best_state, best_ep = -1.0, None, 0
    hist = []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run, t0 = 0.0, time.time()
        for imgs, targets, tlens, _ in tl:
            imgs, targets = imgs.to(device), targets.to(device)
            logp = model(imgs).log_softmax(2)
            T, B, _ = logp.shape
            inlen = torch.full((B,), T, dtype=torch.long)
            loss = crit(logp, targets, inlen, tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
        lr_actual = opt.param_groups[0]["lr"]
        sch.step()

        p, t = predict(model, vl, device)
        d = diagnostico(p, t)
        dt = time.time() - t0
        eta = dt * (args.epochs - ep)
        marca = ""
        if d["exact_match"] >= best_val:
            best_val, best_ep = d["exact_match"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marca = "  <- mejor"
        log("  %-5s %-8.4f %-9.4f %-8.4f %-9.4f %-8.2g %-8.0f %s%s"
            % ("%d/%d" % (ep, args.epochs), run / max(1, len(tl)), d["exact_match"],
               d["cer"], d["len_err"], lr_actual, dt, hhmm(eta), marca))
        hist.append({"ep": ep, "loss": round(run / max(1, len(tl)), 4),
                     "val_exact": d["exact_match"], "val_cer": d["cer"],
                     "val_len_err": round(d["len_err"], 5)})

        if args.muestras and (ep % args.muestras == 0 or ep == args.epochs):
            fallos = [(a, b) for a, b in zip(p, t) if a != b][:6]
            if fallos:
                log("        fallos: " + "  ".join("%s->%s" % (b, a or "(vacio)") for a, b in fallos))

    model.load_state_dict(best_state)
    gp, gt = predict(model, gl, device)
    m = diagnostico(gp, gt)
    ms = latency_ms(model, h, w, device)

    ck = os.path.join(args.out, "ckpt")
    os.makedirs(ck, exist_ok=True)
    torch.save({"model": name, "input_size": (h, w), "num_classes": NUM_CLASSES,
                "state_dict": best_state, "val_exact": best_val},
               os.path.join(ck, "%s_%dx%d_s%d.pt" % (name, h, w, args.seed)))

    log("  mejor epoca %d (val %.4f)" % (best_ep, best_val))
    log("  GOLD-B  exacta %.4f  CER %.4f  len!=6 %.4f (%d cortas, %d largas)  %.2f ms/img  total %s"
        % (m["exact_match"], m["cer"], m["len_err"], m["cortas"], m["largas"],
           ms, hhmm(time.time() - t_start)))
    log.close()

    return {
        "modelo": name, "alto": h, "ancho": w, "aspecto": round(w / h, 3),
        "T": seq_len(name, w), "params": n_par, "params_M": round(n_par / 1e6, 3),
        "semilla": args.seed, "mejor_epoca": best_ep,
        "val_exact": round(best_val, 5),
        "gold_exact": round(m["exact_match"], 5), "gold_cer": round(m["cer"], 5),
        "gold_len_err": round(m["len_err"], 5),
        "gold_cortas": m["cortas"], "gold_largas": m["largas"],
        "ms_img": round(ms, 3),
        "minutos_entrenamiento": round((time.time() - t_start) / 60, 1),
        "historial": hist,
        "pred": gp, "truth": gt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BENCH, "data_sub"),
                    help="carpeta con rec_labels_train.tsv y rec_labels_val.tsv")
    ap.add_argument("--gold", default=os.path.join(BENCH, "data", "rec_labels_gold2.tsv"),
                    help="Gold-B, evaluacion principal")
    ap.add_argument("--out", default=os.path.join(HERE, "resultados"))
    ap.add_argument("--models", default="lprnet,crnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="recorta el train (pruebas)")
    ap.add_argument("--only", default="", help="p.ej. 48x96,64x128")
    ap.add_argument("--muestras", type=int, default=5,
                    help="cada cuantas epocas imprimir fallos de ejemplo (0 desactiva)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoca, 2000 recortes, solo para comprobar que corre")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.smoke:
        a.epochs, a.limit, a.workers = 1, 2000, 0

    res = RESOLUCIONES
    if a.only:
        want = {s.strip().lower() for s in a.only.split(",")}
        res = [r for r in res if ("%dx%d" % (r[0], r[1])) in want]
        if not res:
            raise SystemExit("--only no coincide con ninguna resolucion")

    os.makedirs(a.out, exist_ok=True)
    modelos = [m.strip() for m in a.models.split(",") if m.strip()]
    total = len(modelos) * len(res)
    print("dispositivo %s | %d configuraciones | %d epocas | semilla %d"
          % (a.device, total, a.epochs, a.seed), flush=True)

    dest = os.path.join(a.out, "ablacion_resolucion_s%d.json" % a.seed)
    filas, i, t0 = [], 0, time.time()
    for name in modelos:
        for (h, w, etiqueta) in res:
            i += 1
            transcurrido = time.time() - t0
            resto = (transcurrido / (i - 1) * (total - i + 1)) if i > 1 else 0
            print("\n" + "=" * 78, flush=True)
            print("[%d/%d]  transcurrido %s  |  faltan aprox. %s"
                  % (i, total, hhmm(transcurrido), hhmm(resto) if resto else "?"), flush=True)
            print("=" * 78, flush=True)
            fila = train_one(name, h, w, a, a.device, etiqueta)
            fila["etiqueta"] = etiqueta
            filas.append(fila)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(filas, f, ensure_ascii=False, indent=1)
            print("  guardado en %s" % dest, flush=True)

            print("\n  acumulado hasta ahora:", flush=True)
            print("  %-7s %-10s %-8s %-8s %-9s %-8s"
                  % ("modelo", "entrada", "Gold-B", "CER", "len!=6", "ms/img"), flush=True)
            for r in sorted(filas, key=lambda r: -r["gold_exact"]):
                print("  %-7s %3dx%-6d %-8.4f %-8.4f %-9.4f %-8.2f"
                      % (r["modelo"], r["alto"], r["ancho"], r["gold_exact"],
                         r["gold_cer"], r["gold_len_err"], r["ms_img"]), flush=True)

    print("\ncompletado en %s. Resultados en %s" % (hhmm(time.time() - t0), dest), flush=True)
    print("Siguiente paso:  python ablation_resolucion/report_ablation.py", flush=True)


if __name__ == "__main__":
    main()
