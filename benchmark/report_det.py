#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabla comparativa (HTML) del benchmark de DETECCION desde det_results/*.json."""
import os, json, glob, base64, argparse, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="det_results")
    ap.add_argument("--out", default="det_report.html")
    a = ap.parse_args()
    items = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(a.results, "*.json")))]
    if not items:
        raise SystemExit("no hay resultados en %s (corre det_benchmark.py primero)" % a.results)
    items.sort(key=lambda x: -x["map50_95"])

    names = [i["name"] for i in items]
    m5095 = [i["map50_95"] * 100 for i in items]
    m50 = [i["map50"] * 100 for i in items]
    spd = [i["ms_per_img"] for i in items]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].barh(names[::-1], m5095[::-1], color="#2e74b5"); ax[0].set_title("mAP@50-95 (%)"); ax[0].set_xlim(0, 100)
    for y, v in enumerate(m5095[::-1]): ax[0].text(v, y, " %.1f" % v, va="center", fontsize=9)
    ax[1].barh(names[::-1], m50[::-1], color="#2f9e44"); ax[1].set_title("mAP@50 (%)"); ax[1].set_xlim(0, 100)
    for y, v in enumerate(m50[::-1]): ax[1].text(v, y, " %.1f" % v, va="center", fontsize=9)
    ax[2].barh(names[::-1], spd[::-1], color="#e8590c"); ax[2].set_title("Inferencia (ms/img, menor mejor)")
    for y, v in enumerate(spd[::-1]): ax[2].text(v, y, " %.2f" % v, va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(a.results, "_fig_det.png"), dpi=110); plt.close(fig)
    b64 = "data:image/png;base64," + base64.b64encode(open(os.path.join(a.results, "_fig_det.png"), "rb").read()).decode()

    rows = "".join(
        "<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%.4f</td><td>%.4f</td><td>%.2f</td><td>%.1f</td></tr>" %
        (i["name"], i["map50_95"] * 100, i["map50"] * 100, i["precision"], i["recall"], i["ms_per_img"], i["train_min"])
        for i in items)
    html = """<!DOCTYPE html><html lang=es><head><meta charset=UTF-8>
<meta name=viewport content="width=device-width, initial-scale=1.0"><title>Benchmark deteccion de placas</title>
<style>body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f5f7fa;color:#1f2933;line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding:34px 22px 70px}
.hero{background:linear-gradient(135deg,#1f4e79,#2e74b5);color:#fff;border-radius:16px;padding:28px 32px;margin-bottom:22px}
.hero h1{margin:0 0 6px;font-size:24px}.hero p{margin:0;color:#dfecf8;font-size:14px}
section{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:8px 26px 20px;margin-bottom:18px;box-shadow:0 6px 20px rgba(16,42,67,.06)}
h2{color:#1f4e79;font-size:19px;border-bottom:2px solid #eef4fb;padding-bottom:8px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px;border-radius:9px;overflow:hidden;box-shadow:0 0 0 1px #e2e8f0}
th{background:#1f4e79;color:#fff;text-align:left;padding:9px 12px}td{padding:8px 12px;border-top:1px solid #e2e8f0}
tbody tr:nth-child(even){background:#f7fafd}tbody tr:first-child{background:#eefbf1;font-weight:600}
img{width:100%;border:1px solid #e2e8f0;border-radius:10px;margin:10px 0}
.note{border-left:4px solid #2e74b5;background:#eef4fb;padding:12px 15px;border-radius:0 10px 10px 0;font-size:14px}
</style></head><body><div class=wrap>
<div class=hero><h1>Benchmark de deteccion de placas</h1>
<p>Evaluado sobre el split TEST (260 imagenes, anotado a mano en Roboflow) &middot; mAP, precision, recall &middot; generado __GEN__</p></div>
<section><h2>Comparativa</h2>
<img src="__IMG__">
<table><thead><tr><th>Modelo</th><th>mAP@50-95</th><th>mAP@50</th><th>Precision</th><th>Recall</th><th>ms/img</th><th>Train (min)</th></tr></thead>
<tbody>__ROWS__</tbody></table>
<div class=note>Todos los modelos se entrenan sobre el mismo train (1820) / val (520) y se evaluan sobre el mismo TEST held-out (260). Una sola clase (plate); poligonos Roboflow convertidos a caja axis-aligned. Velocidad medida en la misma GPU.</div>
</section></div></body></html>"""
    html = html.replace("__GEN__", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")).replace("__IMG__", b64).replace("__ROWS__", rows)
    open(a.out, "w", encoding="utf-8").write(html)
    print("escrito", a.out, "con", len(items), "modelos")


if __name__ == "__main__":
    main()
