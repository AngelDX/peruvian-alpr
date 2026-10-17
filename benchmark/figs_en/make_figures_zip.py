#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empaqueta las figuras del manuscrito para el envio a Sensors (MDPI).

Requisitos de la revista: TIFF, PNG, EPS o PDF; minimo 1100 px de ancho y 300 dpi.
Este script verifica ambas cosas antes de comprimir y avisa de lo que falte.

Uso (desde benchmark/figs_en):
    python make_figures_zip.py
"""
import os, sys, zipfile, argparse, shutil
from PIL import Image

# Rutas resueltas desde la ubicacion del script, no desde el directorio de trabajo.
_AQUI = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_AQUI)
_TEX = os.path.abspath(os.path.join(_BENCH, "..", "docs", "paper_sensors", "tex"))


def _b(*p):
    return os.path.join(_BENCH, *p)


def _t(nombre):
    return os.path.join(_TEX, nombre)



# Orden de aparicion en el manuscrito -> fichero fuente (sin extension)
FIGURAS = [
    (1, "fig2_system_en",         "System architecture"),
    (2, "fig1_plate_types_en",    "Peruvian plate types under real capture conditions"),
    (3, "fig_coverage_en",        "Seasonal and diurnal coverage of the corpus"),
    (4, "fig3_recognizers_en",    "Architectures of the two recognizers at 32x100"),
    (5, "fig6_posthoc_boxes_en",  "Post hoc tests on Gold-B, paired bootstrap"),
    (6, "fig5_profiles_en",       "Accuracy and subgroup profiles on Gold-B"),
    (7, "fig4_errors_en",         "Representative recognition errors"),
]
MIN_ANCHO, MIN_DPI = 1100, 299.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=_TEX)
    ap.add_argument("--out", default=os.path.join(_TEX, "..", "Sensors_Figures.zip"))
    ap.add_argument("--stage", default=None, help="carpeta intermedia (por defecto, temporal)")
    a = ap.parse_args()

    tex = os.path.abspath(a.tex)
    stage = os.path.abspath(a.stage or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                                    "_figuras_tmp"))
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)

    incluidas, faltan, avisos = [], [], []
    manifiesto = ["Figures for the manuscript submitted to Sensors (MDPI)",
                  "Peruvian License Plate Recognition at the Edge: Compact LPRNet and CRNN",
                  "Recognizers Distilled from a Rule-Corrected Generic OCR", "",
                  "All raster files are PNG at 300 dpi; vector PDF is supplied where the figure",
                  "is drawn rather than photographic.", ""]

    for n, base, titulo in FIGURAS:
        png = os.path.join(tex, base + ".png")
        if not os.path.exists(png):
            faltan.append((n, base))
            continue
        im = Image.open(png)
        ancho, alto = im.size
        dpi = im.info.get("dpi", (0, 0))[0]
        if ancho < MIN_ANCHO:
            avisos.append("Figure %d: solo %d px de ancho" % (n, ancho))
        if dpi < MIN_DPI:
            # Se corrige la metadata sin remuestrear: los pixeles son los que son.
            im.save(os.path.join(stage, "Figure%d.png" % n), dpi=(300, 300))
            avisos.append("Figure %d: dpi %s -> reescrito a 300 sin remuestrear" % (n, dpi))
        else:
            shutil.copy2(png, os.path.join(stage, "Figure%d.png" % n))
        pdf = os.path.join(tex, base + ".pdf")
        vect = ""
        if os.path.exists(pdf):
            shutil.copy2(pdf, os.path.join(stage, "Figure%d.pdf" % n))
            vect = " + vector PDF"
        incluidas.append(n)
        manifiesto.append("Figure %d.  %s" % (n, titulo))
        manifiesto.append("           Figure%d.png  %d x %d px, 300 dpi%s" % (n, ancho, alto, vect))

    if faltan:
        manifiesto.append("")
        manifiesto.append("MISSING: " + ", ".join("Figure %d (%s)" % f for f in faltan))

    with open(os.path.join(stage, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifiesto) + "\n")

    destino = os.path.abspath(a.out)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(stage)):
            z.write(os.path.join(stage, f), f)
    shutil.rmtree(stage, ignore_errors=True)

    print("zip ->", destino)
    print("figuras incluidas:", incluidas)
    for x in avisos:
        print("  aviso:", x)
    if faltan:
        print("\nFALTAN, y el zip esta incompleto:")
        for n, base in faltan:
            print("  Figure %d (%s.png). Generala con:" % (n, base))
            print("      python figs_en/fig_errors_en.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
