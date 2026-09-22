#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrena LPRNet o CRNN (perdida CTC) para reconocimiento de placas.

Uso:
  python train.py --model lprnet --data data --epochs 60 --batch 256 --out ckpt/lprnet.pt
  python train.py --model crnn   --data data --epochs 60 --batch 256 --out ckpt/crnn.pt
"""
import os, time, argparse, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import PlateDataset, collate_ctc
from models import build_model
from bench_utils import NUM_CLASSES, ctc_greedy_decode


def greedy(logits):
    idx = logits.argmax(2).permute(1, 0)          # (B, T)
    return [ctc_greedy_decode(r.tolist()) for r in idx]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ok = tot = 0
    for imgs, _, _, plates in loader:
        out = model(imgs.to(device))
        for p, t in zip(greedy(out.cpu()), plates):
            ok += (p == t); tot += 1
    return ok / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lprnet", "crnn"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model = build_model(a.model, NUM_CLASSES).to(a.device)
    H, W = model.input_size
    tr = PlateDataset(os.path.join(a.data, "rec_labels_train.tsv"), H, W, augment=True)
    va = PlateDataset(os.path.join(a.data, "rec_labels_val.tsv"), H, W)
    print("train %d | val %d | device %s | %s (T=?, C=%d)" % (len(tr), len(va), a.device, a.model, NUM_CLASSES))
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate_ctc, drop_last=True)
    vl = DataLoader(va, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_ctc)

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for imgs, targets, tlens, _ in tl:
            imgs = imgs.to(a.device); targets = targets.to(a.device)
            logits = model(imgs)                       # (T,B,C)
            logp = logits.log_softmax(2)
            T, B, _ = logp.shape
            inlen = torch.full((B,), T, dtype=torch.long)
            loss = crit(logp, targets, inlen, tlens)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); run += loss.item()
        sch.step()
        acc = evaluate(model, vl, a.device)
        print("epoch %2d/%d  loss %.3f  val_exact %.4f  (%.1fs)" % (ep, a.epochs, run/max(1,len(tl)), acc, time.time()-t0), flush=True)
        if acc >= best:
            best = acc
            torch.save({"model": a.model, "state_dict": model.state_dict(),
                        "input_size": model.input_size, "num_classes": NUM_CLASSES,
                        "val_exact": acc}, a.out)
    print("mejor val_exact %.4f  ->  %s" % (best, a.out))


if __name__ == "__main__":
    main()
