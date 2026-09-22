#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolida TODOS los lotes etiquetados -> gold combinado (OCR + recorte + tipo).
Salidas: data/rec_labels_gold_all.tsv (limpio, para fine-tuning) y gold_all.csv (todo).
"""
import glob, os, csv
import openpyxl
from bench_utils import norm_plate

SHEETS = "../docs/verificacion_gold5000"
mp = {r["id"]: r["nombre_archivo"] for r in csv.DictReader(open(SHEETS + "/mapping.csv", encoding="utf-8-sig"))}
act = {r["fn"]: r for r in csv.DictReader(open("active_selection.csv", encoding="utf-8-sig"))}

all_rows = []; nlot = 0
for f in sorted(glob.glob(SHEETS + "/lote_*.xlsx")):
    if "BACKUP" in f:
        continue
    ws = openpyxl.load_workbook(f, read_only=True).active
    filled = 0
    for r in ws.iter_rows(min_row=5, values_only=True):
        if not r or not r[0]:
            continue
        gid = r[0]; sug = norm_plate(r[2]); rev = (r[3] or "").strip().upper()
        corr = norm_plate(r[4]); rec = (r[5] or "").strip().upper(); tipo = (r[7] or "").strip().upper() if len(r) > 7 else ""
        if rev == "OK": g = sug
        elif rev == "CORREGIR": g = corr
        elif rev == "ILEGIBLE": g = ""
        elif len(norm_plate(rev)) == 6: g = norm_plate(rev)   # placa puesta en col revisión
        else: g = corr or ""
        if rev or corr or rec:
            filled += 1
        fn = mp.get(gid, ""); a = act.get(fn, {})
        all_rows.append({"id": gid, "fn": fn, "path": a.get("path", ""), "gold": g,
                         "recorte": rec, "tipo": tipo, "rev": rev})
    if filled:
        nlot += 1
        print("  %s: %d etiquetadas" % (os.path.basename(f), filled))

with open("gold_all.csv", "w", newline="", encoding="utf-8-sig") as fo:
    w = csv.writer(fo); w.writerow(["id", "fn", "path", "gold", "recorte", "tipo", "rev"])
    for x in all_rows:
        w.writerow([x["id"], x["fn"], x["path"], x["gold"], x["recorte"], x["tipo"], x["rev"]])

clean = [x for x in all_rows if x["gold"] and len(x["gold"]) == 6 and x["recorte"] != "INCOMPLETA"
         and x["path"] and os.path.isfile(x["path"])]
seen = set(); tsv = []
for x in clean:
    if x["path"] not in seen:
        seen.add(x["path"]); tsv.append((x["path"], x["gold"]))
open("data/rec_labels_gold_all.tsv", "w", encoding="utf-8").write("".join("%s\t%s\n" % t for t in tsv))
from collections import Counter
print("lotes con etiquetas:", nlot)
print("gold total:", len(all_rows), "| limpio p/ fine-tuning:", len(tsv),
      "| INCOMPLETA:", sum(1 for x in all_rows if x["recorte"] == "INCOMPLETA"),
      "| ILEGIBLE:", sum(1 for x in all_rows if x["rev"] == "ILEGIBLE"))
print("tipos:", dict(Counter(x["tipo"] for x in all_rows if x["tipo"]).most_common()))
print("-> data/rec_labels_gold_all.tsv, gold_all.csv")
