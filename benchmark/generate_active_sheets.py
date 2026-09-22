#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera hojas de verificación (Excel) por lotes a partir de active_selection.csv,
para etiquetar fácil verificando con SUNARP. La columna 'placa' viene pre-llenada
con la sugerencia del modelo; el anotador solo corrige las erróneas.

Salida: verificacion_gold5000/lote_XX.xlsx (imagen + placa pre-llenada + motivo + notas)
        + verificacion_gold5000/mapping.csv
Uso: python generate_active_sheets.py --batch 1000
"""
import os, csv, argparse, tempfile
from PIL import Image as PILImage
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

ARIAL = "Arial"


def build_sheet(rows, start_idx, out_path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Verificacion"
    ws["A1"] = "Ampliación del gold — verificación de placas contra SUNARP"
    ws["A1"].font = Font(name=ARIAL, size=13, bold=True, color="1F4E79"); ws.merge_cells("A1:F1")
    ws["A2"] = ("(1) LECTURA en 'revisión': OK (sugerida correcta) / CORREGIR (escriba la lectura correcta en "
                "'placa_correcta', AAA-NNN, verificada con SUNARP; puede inferir aunque esté algo cortada) / "
                "ILEGIBLE (no se puede leer). (2) 'recorte': COMPLETA o INCOMPLETA (le faltan caracteres). "
                "Una placa cortada PERO legible: márquela con su número Y recorte=INCOMPLETA. Filas rojas = 'modelos discrepan'.")
    ws["A2"].font = Font(name=ARIAL, size=10, italic=True); ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:G2"); ws.row_dimensions[2].height = 64

    hdr = 4
    headers = ["id", "imagen_placa", "placa_sugerida", "revisión", "placa_correcta", "recorte", "motivo", "tipo_placa"]
    for col, name in zip("ABCDEFGH", headers):
        c = ws["%s%d" % (col, hdr)]; c.value = name; c.font = Font(name=ARIAL, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E79"); c.alignment = Alignment(horizontal="center", vertical="center")
    widths = {"A": 8, "B": 34, "C": 14, "D": 13, "E": 14, "F": 14, "G": 18, "H": 20}
    for col, w in widths.items(): ws.column_dimensions[col].width = w
    yellow = PatternFill("solid", fgColor="FFFDE7"); green = PatternFill("solid", fgColor="EAF7EE")
    blue = PatternFill("solid", fgColor="E7F0FA")
    warn = PatternFill("solid", fgColor="FDECEA"); thin = Border(*[Side(style="thin", color="DDDDDD")] * 4)
    dv = DataValidation(type="list", formula1='"OK,CORREGIR,ILEGIBLE"', allow_blank=True); ws.add_data_validation(dv)
    dvr = DataValidation(type="list", formula1='"COMPLETA,INCOMPLETA"', allow_blank=True); ws.add_data_validation(dvr)
    dvt = DataValidation(type="list", formula1='"BLANCA,FRANJA_AMARILLA,FRANJA_VERDE,CELESTE,AMARILLA,OTRA,NO_VISIBLE"', allow_blank=True); ws.add_data_validation(dvt)
    purple = PatternFill("solid", fgColor="F3E9F7")

    tmp = tempfile.mkdtemp(); mapping = []
    r0 = hdr + 1
    for i, row in enumerate(rows):
        gid = "G%04d" % (start_idx + i + 1); r = r0 + i
        ws["A%d" % r] = gid; ws["A%d" % r].font = Font(name=ARIAL, bold=True)
        ws["A%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        try:
            im = PILImage.open(row["path"]).convert("RGB")
            sc = 85.0 / im.height; im = im.resize((max(1, int(im.width * sc)), 85), PILImage.LANCZOS)
            p = os.path.join(tmp, gid + ".jpg"); im.save(p, quality=80)
            xi = openpyxl.drawing.image.Image(p); xi.width = im.width; xi.height = im.height
            ws.add_image(xi, "B%d" % r)
        except Exception:
            ws["B%d" % r] = "(imagen no disponible)"
        sp = row["silver"]; disp = (sp[:3] + "-" + sp[3:]) if len(sp) == 6 else sp
        ws["C%d" % r] = disp; ws["C%d" % r].font = Font(name="Consolas", size=13, bold=True)
        ws["C%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        ws["D%d" % r].fill = green; ws["D%d" % r].border = thin; dv.add(ws["D%d" % r])
        ws["D%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        ws["E%d" % r].fill = yellow; ws["E%d" % r].border = thin
        ws["E%d" % r].font = Font(name="Consolas", size=12); ws["E%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        ws["F%d" % r].fill = blue; ws["F%d" % r].border = thin; dvr.add(ws["F%d" % r])
        ws["F%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        ws["G%d" % r] = row["motivo"]; ws["G%d" % r].font = Font(name=ARIAL, size=9)
        ws["G%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        if row["motivo"] == "modelos discrepan": ws["G%d" % r].fill = warn
        ws["H%d" % r].fill = purple; ws["H%d" % r].border = thin; dvt.add(ws["H%d" % r])
        ws["H%d" % r].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[r].height = 68
        mapping.append((gid, row["fn"], row["silver"], row["motivo"]))
    ws.freeze_panes = "A%d" % r0
    wb.save(out_path)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="active_selection.csv")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--out", default="verificacion_gold5000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.sel, encoding="utf-8-sig")))
    print("total a etiquetar:", len(rows))
    allmap = []
    for b in range(0, len(rows), a.batch):
        chunk = rows[b:b + a.batch]; num = b // a.batch + 1
        out = os.path.join(a.out, "lote_%02d.xlsx" % num)
        allmap += build_sheet(chunk, b, out)
        print("  lote_%02d.xlsx  (%d placas)" % (num, len(chunk)))
    with open(os.path.join(a.out, "mapping.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "nombre_archivo", "placa_sugerida", "motivo"]); w.writerows(allmap)
    print("mapping.csv escrito. Lotes en", a.out)


if __name__ == "__main__":
    main()
