#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exporta todas las tablas del benchmark a CSV y LaTeX (booktabs), listas para el
manuscrito. Tambien calcula el desglose por tipo de placa (auto vs moto) por aspecto.

Salida: tables/csv/*.csv  y  tables/latex/*.tex
"""
import os, re, json, csv, glob
import numpy as np

os.makedirs("tables/csv", exist_ok=True)
os.makedirs("tables/latex", exist_ok=True)

DASH = json.load(open("dashboard_data.json", encoding="utf-8"))
SZ = json.load(open("size_analysis.json", encoding="utf-8"))
FS = json.load(open("full_size_distribution.json", encoding="utf-8"))


def esc(s):
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("²", r"\textsuperscript{2}"), ("±", r"$\pm$"), ("↔", r"$\leftrightarrow$"), ("→", r"$\rightarrow$"), ("×", r"$\times$")]:
        s = s.replace(a, b)
    return s


def write_csv(name, header, rows):
    with open("tables/csv/%s.csv" % name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def write_latex(name, header, rows, caption, label, align=None, note=None):
    ncol = len(header)
    align = align or ("l" + "r" * (ncol - 1))
    L = []
    L.append("% requiere \\usepackage{booktabs}")
    L.append("\\begin{table}[htbp]")
    L.append("  \\centering")
    L.append("  \\caption{%s}" % esc(caption))
    L.append("  \\label{%s}" % label)
    L.append("  \\begin{tabular}{%s}" % align)
    L.append("    \\toprule")
    L.append("    " + " & ".join("\\textbf{%s}" % esc(h) for h in header) + " \\\\")
    L.append("    \\midrule")
    for r in rows:
        L.append("    " + " & ".join(esc(c) for c in r) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    if note:
        L.append("  \\\\[2pt] {\\footnotesize %s}" % esc(note))
    L.append("\\end{table}")
    open("tables/latex/%s.tex" % name, "w", encoding="utf-8").write("\n".join(L) + "\n")


def export(name, header, rows, caption, label, align=None, note=None):
    write_csv(name, header, rows)
    write_latex(name, header, rows, caption, label, align, note)
    print("  %-22s %d filas" % (name, len(rows)))


# ---------- T1 Deteccion ----------
det = sorted(DASH["det"], key=lambda x: -x["map50_95"])
export("t1_deteccion",
       ["Modelo", "mAP@50-95", "mAP@50", "Precision", "Recall", "ms/img", "Train (min)"],
       [[d["name"], "%.4f" % d["map50_95"], "%.3f" % d["map50"], "%.4f" % d["precision"],
         "%.3f" % d["recall"], "%.2f" % d["ms_per_img"], "%.0f" % d["train_min"]] for d in det],
       "Benchmark de deteccion de placa sobre el conjunto de test (260 imagenes anotadas a mano). "
       "mAP@50 y recall estan saturados; el diferenciador es mAP@50-95.",
       "tab:deteccion",
       note="Entrenamiento: 1820 imagenes; validacion: 520. Velocidad medida en NVIDIA RTX 5060 Ti.")

# ---------- T2 Reconocimiento ----------
rf = DASH["rec_final"]
recmap = [("PaddleOCR + corrector", "PaddleOCR+corr", "baseline"),
          ("LPRNet", "LPRNet", "entrenado"),
          ("YOLO+CRNN", "YOLO+CRNN", "entrenado"),
          ("PaddleOCR (crudo)", "PaddleOCR", "baseline")]
export("t2_reconocimiento",
       ["Modelo", "Exact-match (%)", "CER", "Aciertos/782", "Tipo"],
       [[disp, "%.2f" % (rf[k]["exact_match"] * 100), "%.4f" % rf[k]["cer"], rf[k]["aciertos"], tipo]
        for disp, k, tipo in recmap],
       "Benchmark de reconocimiento sobre el gold de 782 placas verificadas con SUNARP. "
       "El corrector por posicion (O/0, I/1) no altera a los modelos entrenados.",
       "tab:reconocimiento",
       note="LPRNet y CRNN entrenados sobre 100000 crops silver (etiqueta OCR+corrector).")

# ---------- T3 OCR por tamano ----------
mlabels = [("lprnet", "LPRNet (%)"), ("crnn", "CRNN (%)"),
           ("paddle", "PaddleOCR+corr (%)"), ("paddle_raw", "PaddleOCR crudo (%)")]
rows3 = []
for b in SZ["bins"]:
    rows3.append([b["label"], b["n"]] + ["%.1f" % (b[m] * 100) for m, _ in mlabels])
rows3.append(["Total", SZ["n"]] + ["%.1f" % (SZ["overall"][m] * 100) for m, _ in mlabels])
export("t3_ocr_por_tamano",
       ["Rango area (px2)", "n"] + [lab for _, lab in mlabels],
       rows3,
       "Exact-match por rango de area de placa sobre el gold (782). El OCR generico se degrada en "
       "placas pequenas; los modelos entrenados se mantienen estables.",
       "tab:ocr_tamano",
       note="Correlacion area-acierto: PaddleOCR crudo +%.2f; LPRNet %.2f; CRNN %.2f." %
            (SZ["corr"]["paddle_raw"], SZ["corr"]["lprnet"], SZ["corr"]["crnn"]))

# ---------- T4 Metricas de tamano (valor / interpretacion) ----------
a = SZ["area_stats"]; b0 = SZ["bins"][0]
best = max(SZ["bins"], key=lambda x: x["lprnet"])
rows4 = [
    ["Ancho de placa", "%d-%d px" % tuple(SZ["w_range"]), "Camara a distancia casi fija; poca variacion horizontal."],
    ["Alto de placa", "%d-%d px" % tuple(SZ["h_range"]), "Varia mas, por el angulo e inclinacion del vehiculo."],
    ["Area de placa", "%.1fk-%.1fk px2" % (a["min"]/1000, a["max"]/1000), "Factor x%.1f entre la mas pequena y la mas grande." % (a["max"]/a["min"])],
    ["Corr. area-acierto (PaddleOCR crudo)", "+%.2f" % SZ["corr"]["paddle_raw"], "Dependencia clara del tamano: placas pequenas, mas error."],
    ["Corr. area-acierto (LPRNet)", "%.2f" % SZ["corr"]["lprnet"], "Practicamente independiente del tamano: robusto."],
    ["Peor rango (%s) PaddleOCR crudo" % b0["label"], "%.1f%%" % (b0["paddle_raw"]*100), "El OCR generico colapsa en placas pequenas."],
    ["Peor rango (%s) LPRNet" % b0["label"], "%.1f%%" % (b0["lprnet"]*100), "El modelo entrenado casi triplica al OCR generico."],
    ["Mejor rango (%s) LPRNet" % best["label"], "%.1f%%" % (best["lprnet"]*100), "Optimo en rango medio: la precision no es monotona."],
    ["Tamano minimo recomendado", "~4.0k px2 (95x50)", "Umbral para OCR fiable; reportar en Usage Notes."],
]
export("t4_metricas_tamano",
       ["Metrica", "Valor", "Interpretacion"], rows4,
       "Metricas del efecto del tamano de placa en el reconocimiento, con su interpretacion.",
       "tab:metricas_tamano", align="p{5.4cm}rp{6.2cm}")

# ---------- T5 Distribucion de tamano (dataset completo) ----------
def srow(disp, s):
    fmt = (lambda v: "%.2f" % v) if disp == "Aspecto" else (lambda v: "%.0f" % v)
    return [disp, fmt(s["min"]), fmt(s["p1"]), fmt(s["q1"]), fmt(s["median"]),
            fmt(s["mean"]), fmt(s["q3"]), fmt(s["p99"]), fmt(s["max"]), fmt(s["std"])]
rows5 = [srow("Ancho (px)", FS["ancho_px"]), srow("Alto (px)", FS["alto_px"]),
         srow("Area (px2)", FS["area_px2"]), srow("Aspecto", FS["aspect"])]
export("t5_distribucion_tamano",
       ["Dimension", "min", "p1", "Q1", "Mediana", "Media", "Q3", "p99", "max", "SD"],
       rows5,
       "Distribucion de tamano de placa sobre el dataset completo (%s crops). "
       "El aspecto es bimodal (~2.0 auto, ~1.0 moto/mototaxi)." % f"{FS['n_total']:,}".replace(",", " "),
       "tab:distribucion_tamano",
       note="Tamano derivado de la anotacion silver del detector; splits train/val/test.")

# ---------- T6 Desglose por tipo de placa (proxy por aspecto) ----------
RX_TUPLE = re.compile(r"\((\d+),\s*(\d+),\s*\d+\)")
RX_WXH = re.compile(r"_(\d+)x(\d+)[_.]")
def parse_wh(fn):
    m = RX_TUPLE.search(fn)
    if m: return int(m.group(2)), int(m.group(1))
    m = RX_WXH.search(fn)
    if m: return int(m.group(1)), int(m.group(2))
    return None
Ws, Hs = [], []
for split in ("train", "val", "test"):
    d = os.path.join("../Peruvian_ALPR_Dataset_final/crops", split)
    if not os.path.isdir(d): continue
    with os.scandir(d) as it:
        for e in it:
            if not e.name.lower().endswith(".jpg"): continue
            wh = parse_wh(e.name)
            if wh: Ws.append(wh[0]); Hs.append(wh[1])
Ws = np.array(Ws); Hs = np.array(Hs); Ar = Ws / np.maximum(Hs, 1); Area = Ws * Hs
THR = 1.5
groups = [("Estandar (auto)", Ar >= THR), ("Cuadrada (moto/mototaxi)", Ar < THR)]
rows6 = []
for name, mask in groups:
    n = int(mask.sum())
    rows6.append([name, n, "%.1f" % (100*n/len(Ar)),
                  "%d" % np.median(Ws[mask]), "%d" % np.median(Hs[mask]),
                  "%d" % np.median(Area[mask]), "%.2f" % np.median(Ar[mask])])
export("t6_tipo_placa",
       ["Tipo (proxy aspecto)", "n", "%", "Ancho med.", "Alto med.", "Area med.", "Aspecto med."],
       rows6,
       "Desglose del dataset por tipo de placa, estimado con un umbral de aspecto (%.1f). "
       "Proxy no verificado: separa placas de una fila (auto) de las cuadradas (moto/mototaxi)." % THR,
       "tab:tipo_placa",
       note="Clasificacion por relacion de aspecto ancho/alto; no verificada manualmente.")

print("\nTablas exportadas en tables/csv/ y tables/latex/")
print("Tipo de placa: estandar=%d (%.1f%%)  cuadrada=%d (%.1f%%)" %
      (rows6[0][1], float(rows6[0][2]), rows6[1][1], float(rows6[1][2])))
