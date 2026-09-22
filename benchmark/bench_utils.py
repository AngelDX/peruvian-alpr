# -*- coding: utf-8 -*-
"""Utilidades comunes del benchmark: charset, codificacion CTC, metricas."""
import re
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"      # 36 simbolos
BLANK = 0                                            # indice CTC blank
C2I = {c: i + 1 for i, c in enumerate(CHARS)}        # char -> idx (1..36)
I2C = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1                          # + blank

def norm_plate(p):
    """Placa a 6 alfanumericos en mayuscula (sin guion). '' si no es valida."""
    p = re.sub(r"[^A-Za-z0-9]", "", str(p).upper())
    return p if len(p) == 6 and all(c in CHARS for c in p) else ""

def encode(plate):
    return [C2I[c] for c in plate]

def ctc_greedy_decode(seq_idx):
    """seq_idx: lista de indices por timestep -> string (colapsa repetidos y blanks)."""
    out = []
    prev = -1
    for idx in seq_idx:
        if idx != prev and idx != BLANK:
            out.append(I2C.get(int(idx), ""))
        prev = idx
    return "".join(out)

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]

def metrics(preds, truths):
    """Devuelve dict con exact-match y CER."""
    n = len(truths)
    em = sum(1 for p, t in zip(preds, truths) if p == t)
    ed = sum(levenshtein(p, t) for p, t in zip(preds, truths))
    tl = sum(len(t) for t in truths)
    return {"n": n, "exact_match": round(em / n, 4) if n else 0.0,
            "cer": round(ed / tl, 4) if tl else 0.0, "aciertos": em}
