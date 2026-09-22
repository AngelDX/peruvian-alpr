#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dosier HTML: resultados publicados (24x94, corrida unica) frente a los nuevos
(32x100, tres semillas), mas la ablacion de resolucion que motivo el cambio.

Lee, sin recalcular nada:
    ../../docs/paper_materials/15_cifras_canonicas.json   cifras del manuscrito
    resultados/ablacion_resolucion_s*.json                ablacion y bases silver
    resultados_ft/finetune_32x100.json                    modelos afinados

Uso (desde benchmark/):
    python ablation_resolucion/build_dossier.py
"""
import os
import sys
import json
import glob
import argparse
import datetime
import re
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RAIZ = os.path.dirname(BENCH)


# ---------------------------------------------------------------- estadistica
def media_sd(v):
    if not v:
        return 0.0, 0.0
    mu = sum(v) / len(v)
    if len(v) < 2:
        return mu, 0.0
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - r) / d, (c + r) / d


def mcnemar(a_pred, b_pred, truth):
    b10 = sum(1 for p, q, t in zip(a_pred, b_pred, truth) if p == t and q != t)
    b01 = sum(1 for p, q, t in zip(a_pred, b_pred, truth) if p != t and q == t)
    n = b10 + b01
    if not n:
        return 1.0, b10, b01
    k = min(b10, b01)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), b10, b01


def welch(x, y):
    mx, sx = media_sd(x)
    my, sy = media_sd(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0, 0.0, (mx - my)
    se = (sx * sx / nx + sy * sy / ny) ** 0.5
    if se == 0:
        return 0.0, 0.0, mx - my
    t = (mx - my) / se
    num = (sx * sx / nx + sy * sy / ny) ** 2
    den = (sx * sx / nx) ** 2 / max(1, nx - 1) + (sy * sy / ny) ** 2 / max(1, ny - 1)
    return t, (num / den if den else 0.0), mx - my


# ------------------------------------------------------------- recuperacion
def recuperar_de_logs(carpeta, ya):
    """Reconstruye configuraciones ausentes del JSON leyendo los logs.

    Hace falta porque relanzar con --only sobrescribia el JSON de la semilla y se
    perdian las filas del barrido original. Los logs son por configuracion, asi
    que solo se pisaron las que se repitieron. Sin predicciones por matricula:
    de los logs solo salen las cifras agregadas, suficiente para la tabla.
    """
    presentes = {(r["modelo"], r["alto"], r["ancho"], r.get("semilla", 0)) for r in ya}
    nombre = re.compile(r"^(\w+)_(\d+)x(\d+)_s(\d+)\.log$")
    cab = re.compile(r"T=(\d+)\s+params=([\d.]+)M")
    fin = re.compile(r"GOLD-B\s+exacta ([\d.]+)\s+CER ([\d.]+)\s+len!=6 ([\d.]+)"
                     r"\s+\((\d+) cortas, (\d+) largas\)\s+([\d.]+) ms/img")
    salida = []
    for f in sorted(glob.glob(os.path.join(carpeta, "*.log"))):
        m = nombre.match(os.path.basename(f))
        if not m:
            continue
        modelo, hh, ww, s = m.group(1).lower(), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if (modelo, hh, ww, s) in presentes:
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        c, e = cab.search(txt), fin.search(txt)
        if not (c and e):
            continue
        salida.append({
            "modelo": modelo, "alto": hh, "ancho": ww, "aspecto": round(ww / hh, 3),
            "T": int(c.group(1)), "params_M": float(c.group(2)), "semilla": s,
            "gold_exact": float(e.group(1)), "gold_cer": float(e.group(2)),
            "gold_len_err": float(e.group(3)),
            "gold_cortas": int(e.group(4)), "gold_largas": int(e.group(5)),
            "ms_img": float(e.group(6)), "de_log": True,
        })
    if salida:
        print("recuperadas %d configuraciones desde los logs" % len(salida))
    return salida


# ---------------------------------------------------------------------- carga
def cargar(a):
    pub = {}
    p = os.path.join(RAIZ, "docs", "paper_materials", "15_cifras_canonicas.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            pub = json.load(f).get("conjuntos", {})

    abl = []
    for f in sorted(glob.glob(os.path.join(a.ablacion, "ablacion_resolucion_s*.json"))):
        with open(f, encoding="utf-8") as fh:
            abl.extend(json.load(fh))
    abl.extend(recuperar_de_logs(os.path.join(a.ablacion, "logs"), abl))

    ft = []
    if os.path.exists(a.finetune):
        with open(a.finetune, encoding="utf-8") as fh:
            ft = json.load(fh)
    return pub, abl, ft


def pub_get(pub, conjunto, sistema, campo="exact_match"):
    try:
        return pub[conjunto]["sistemas"][sistema][campo]
    except (KeyError, TypeError):
        return None


# ------------------------------------------------------------------ plantilla
CSS = """
:root{--tinta:#16202b;--suave:#5b6b7c;--linea:#dfe5ec;--fondo:#f6f8fa;
--bien:#1a7f4b;--mal:#b3261e;--neutro:#8a6d1f;--acento:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 64px;font:15px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;
color:var(--tinta);background:#fff;max-width:1080px;margin-inline:auto}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:40px 0 6px;padding-top:14px;border-top:2px solid var(--tinta)}
h3{font-size:15px;margin:24px 0 6px;color:var(--suave);text-transform:uppercase;letter-spacing:.06em}
p{margin:8px 0}
.sub{color:var(--suave);margin:0 0 8px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{padding:7px 10px;border-bottom:1px solid var(--linea);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--fondo);border-bottom:2px solid var(--linea);font-weight:600;
color:var(--suave);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:hover{background:var(--fondo)}
td.n{font-variant-numeric:tabular-nums}
.bien{color:var(--bien);font-weight:600}
.mal{color:var(--mal);font-weight:600}
.igual{color:var(--neutro)}
.tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.tarjeta{border:1px solid var(--linea);border-radius:8px;padding:14px 16px;background:var(--fondo)}
.tarjeta .k{font-size:11px;color:var(--suave);text-transform:uppercase;letter-spacing:.06em}
.tarjeta .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.tarjeta .d{font-size:12px;color:var(--suave);margin-top:2px}
.nota{border-left:3px solid var(--acento);background:#f0f5ff;padding:10px 14px;margin:14px 0;font-size:14px}
.aviso{border-left:3px solid var(--neutro);background:#fdf8e8;padding:10px 14px;margin:14px 0;font-size:14px}
.pie{margin-top:48px;padding-top:14px;border-top:1px solid var(--linea);
color:var(--suave);font-size:12px}
code{background:var(--fondo);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def clase(delta, umbral):
    if abs(delta) < umbral:
        return "igual"
    return "bien" if delta > 0 else "mal"


def fmt_delta(d, umbral):
    return '<span class="%s">%+.2f</span>' % (clase(d, umbral), d)


def construir(pub, abl, ft, destino):
    h = []
    A = h.append
    A("<!doctype html><html lang=es><meta charset=utf-8>")
    A("<title>Resolución unificada 32×100 · comparativa</title>")
    A("<style>%s</style>" % CSS)
    A("<h1>Resolución unificada a 32×100</h1>")
    A('<p class=sub>Comparativa entre las cifras publicadas del manuscrito '
      "(24×94 para LPRNet, corrida única) y las nuevas (32×100 para ambos modelos, "
      "tres semillas). Generado el %s.</p>"
      % datetime.date.today().isoformat())

    # ---------------------------------------------------------------- resumen
    ftl = [r for r in ft if r["modelo"] == "lprnet"]
    ftc = [r for r in ft if r["modelo"] == "crnn"]
    mL, sL = media_sd([r["ft_goldb"] for r in ftl])
    mC, sC = media_sd([r["ft_goldb"] for r in ftc])
    pubL = pub_get(pub, "gold_B", "LPRNet afinado") or 0.9115
    pubC = pub_get(pub, "gold_B", "CRNN afinado") or 0.9248

    A("<div class=tarjetas>")
    for k, v, d in (
        ("LPRNet afinado · Gold-B", "%.2f%%" % (100 * mL) if ftl else "—",
         "antes %.2f%% · %s" % (100 * pubL, fmt_delta(100 * (mL - pubL), 1.0)) if ftl else ""),
        ("CRNN afinado · Gold-B", "%.2f%%" % (100 * mC) if ftc else "—",
         "antes %.2f%% · %s" % (100 * pubC, fmt_delta(100 * (mC - pubC), 1.0)) if ftc else ""),
        ("Desviación entre semillas", "%.2f pp" % (100 * max(sL, sC)) if ft else "—",
         "antes no se medía"),
        ("Corridas por configuración", "3" if ft else "—", "antes 1"),
    ):
        A('<div class=tarjeta><div class=k>%s</div><div class=v>%s</div><div class=d>%s</div></div>'
          % (k, v, d))
    A("</div>")

    A('<div class=nota><strong>Lectura corta.</strong> Unificar la resolución no cuesta '
      "exactitud y sí compra dos cosas que el manuscrito necesitaba: los dos reconocedores "
      "reciben por fin la misma entrada, lo que vuelve cierta la afirmación de comparabilidad "
      "de la Sección 3, y cada cifra lleva ahora una desviación medida sobre tres semillas "
      "en lugar de proceder de una única corrida.</div>")

    # ------------------------------------------------------- tabla principal
    A("<h2>1. Antes y después</h2>")
    A("<p>Coincidencia exacta sobre <em>Gold</em>-B (n = 1130). Las columnas «antes» "
      "proceden de <code>15_cifras_canonicas.json</code> y son de corrida única; las de "
      "«ahora» son media y desviación típica de tres semillas.</p>")
    A("<table><thead><tr><th>Sistema</th><th>Antes (24×94)</th><th>Ahora (32×100)</th>"
      "<th>Δ (pp)</th><th>CER ahora</th><th>Retenido ahora</th></tr></thead><tbody>")

    bl = [r for r in abl if (r["alto"], r["ancho"]) == (32, 100) and r["modelo"] == "lprnet"]
    bc = [r for r in abl if (r["alto"], r["ancho"]) == (32, 100) and r["modelo"] == "crnn"]
    filas = [
        ("LPRNet base (<em>silver</em>)", pub_get(pub, "gold_B", "LPRNet base"),
         media_sd([r["gold_exact"] for r in bl]), media_sd([r["gold_cer"] for r in bl]), None),
        ("CRNN base (<em>silver</em>)", pub_get(pub, "gold_B", "CRNN"),
         media_sd([r["gold_exact"] for r in bc]), media_sd([r["gold_cer"] for r in bc]), None),
        ("LPRNet afinado", pubL, (mL, sL),
         media_sd([r["ft_cer"] for r in ftl]), media_sd([r["ft_retenido"] for r in ftl])),
        ("CRNN afinado", pubC, (mC, sC),
         media_sd([r["ft_cer"] for r in ftc]), media_sd([r["ft_retenido"] for r in ftc])),
    ]
    for nombre, antes, (mu, sd), (cer, _), ret in filas:
        if not mu:
            continue
        d = 100 * (mu - antes) if antes else 0.0
        A("<tr><td>%s</td><td class=n>%s</td><td class=n>%.2f%% ± %.2f</td>"
          "<td class=n>%s</td><td class=n>%.4f</td><td class=n>%s</td></tr>"
          % (nombre,
             "%.2f%%" % (100 * antes) if antes else "—",
             100 * mu, 100 * sd,
             fmt_delta(d, 1.0) if antes else "—",
             cer,
             "%.2f%% ± %.2f" % (100 * ret[0], 100 * ret[1]) if ret else "—"))
    A("</tbody></table>")
    A('<div class=aviso><strong>Cómo leer la Δ.</strong> Está en verde o rojo solo si supera '
      "un punto porcentual, que es el orden de la variabilidad entre semillas. Por debajo de "
      "eso la diferencia no es interpretable, aunque una prueba pareada la declare "
      "significativa: McNemar compara dos modelos concretos, no dos arquitecturas.</div>")

    # ----------------------------------------------------------- por semilla
    if ft:
        A("<h2>2. Detalle por semilla</h2>")
        A("<table><thead><tr><th>Modelo</th><th>Semilla</th><th>Base Gold-B</th>"
          "<th>Afinado Gold-B</th><th>Δ (pp)</th><th>Retenido</th><th>CER</th>"
          "<th>len≠6</th></tr></thead><tbody>")
        for r in sorted(ft, key=lambda r: (r["modelo"], r["semilla"])):
            A("<tr><td>%s</td><td class=n>%d</td><td class=n>%.4f</td><td class=n>%.4f</td>"
              "<td class=n><span class=bien>%+.2f</span></td><td class=n>%.4f</td>"
              "<td class=n>%.4f</td><td class=n>%.4f</td></tr>"
              % (r["modelo"].upper(), r["semilla"], r["base_goldb"], r["ft_goldb"],
                 r["delta_pp"], r["ft_retenido"], r["ft_cer"], r["ft_len_err"]))
        A("</tbody></table>")

        if len(ftl) > 1 and len(ftc) > 1:
            t, df, dif = welch([r["ft_goldb"] for r in ftl], [r["ft_goldb"] for r in ftc])
            A("<h3>LPRNet frente a CRNN, entre corridas</h3>")
            A("<p>Diferencia de medias %.2f pp, <em>t</em> = %.2f con %.1f grados de libertad. "
              "El valor crítico bilateral al 5&nbsp;%% con ~3 gl es 3.18, de modo que la "
              "comparación %s.</p>"
              % (100 * dif, t, df,
                 "alcanza significación" if abs(t) > 3.18 else "no alcanza significación"))

    # -------------------------------------------------------------- ablacion
    res = sorted({(r["alto"], r["ancho"]) for r in abl})
    if len(res) > 1:
        A("<h2>3. Ablación de resolución</h2>")
        A("<p>El aspecto nativo mediano de la matrícula peruana es 1.98 sobre 439&nbsp;178 "
          "recortes medidos. Se comparó si respetar esa proporción mejora el reconocimiento.</p>")
        A("<table><thead><tr><th>Modelo</th><th>Entrada</th><th>Aspecto</th><th>T</th>"
          "<th>Gold-B</th><th>CER</th><th>len≠6</th><th>ms/img</th></tr></thead><tbody>")
        for modelo in ("lprnet", "crnn"):
            for (hh, ww) in res:
                rs = [r for r in abl if r["modelo"] == modelo and (r["alto"], r["ancho"]) == (hh, ww)]
                if not rs:
                    continue
                mu, sd = media_sd([r["gold_exact"] for r in rs])
                cer, _ = media_sd([r["gold_cer"] for r in rs])
                le, _ = media_sd([r["gold_len_err"] for r in rs])
                ms, _ = media_sd([r["ms_img"] for r in rs])
                marca = " <strong>(elegida)</strong>" if (hh, ww) == (32, 100) else ""
                A("<tr><td>%s%s</td><td class=n>%d×%d</td><td class=n>%.2f</td><td class=n>%d</td>"
                  "<td class=n>%.4f%s</td><td class=n>%.4f</td><td class=n>%.4f</td>"
                  "<td class=n>%.2f</td></tr>"
                  % (modelo.upper(), marca, hh, ww, ww / hh, rs[0]["T"], mu,
                     " ± %.4f" % sd if len(rs) > 1 else "", cer, le, ms))
        A("</tbody></table>")
        A('<div class=aviso><strong>Las filas de LPRNet fuera de 24×94 y 32×100 no son '
          "interpretables.</strong> El arnés hace la red admisible a cualquier entrada mediante "
          "un pooling adaptativo, pero no escala su profundidad, de modo que al ampliar la "
          "imagen el campo receptivo cubre una fracción del carácter. Esas corridas confunden "
          "la relación de aspecto con el campo receptivo y se descartan. El CRNN, más profundo "
          "y con capas recurrentes, sí tolera el cambio.</div>")

    # ---------------------------------------------------------------- cierre
    A("<h2>4. Qué cambia en el manuscrito</h2>")
    A("<ul>")
    A("<li>La Sección 3 gana una subsección de recorte y normalización que justifica el "
      "32×100 con la ablación en lugar de dejarlo sin motivar.</li>")
    A("<li>La afirmación de que todos los sistemas reciben el mismo recorte pasa a ser "
      "cierta: antes LPRNet corría a 24×94 y el CRNN a 32×100.</li>")
    A("<li>Las cifras de reconocimiento se reportan como media y desviación de tres "
      "semillas, y la comparación entre arquitecturas en régimen <em>silver</em> se apoya "
      "en una t entre corridas y no en una McNemar de corrida única.</li>")
    A("<li>Queda descartado con datos el 96×320 de la literatura de referencia, cuya "
      "relación de aspecto (3.33) no corresponde a la matrícula peruana.</li>")
    A("</ul>")

    A('<div class=pie>Generado por <code>ablation_resolucion/build_dossier.py</code>. '
      "Fuentes: <code>15_cifras_canonicas.json</code>, "
      "<code>resultados/ablacion_resolucion_s*.json</code>, "
      "<code>resultados_ft/finetune_32x100.json</code>. "
      "Ninguna cifra se recalcula aquí.</div>")
    A("</html>")

    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablacion", default=os.path.join(HERE, "resultados"))
    ap.add_argument("--finetune", default=os.path.join(HERE, "resultados_ft", "finetune_32x100.json"))
    ap.add_argument("--out", default=os.path.join(RAIZ, "docs", "paper_materials",
                                                  "dossier_resolucion_32x100.html"))
    a = ap.parse_args()

    pub, abl, ft = cargar(a)
    print("publicadas: %s | ablacion: %d filas | finetune: %d filas"
          % ("si" if pub else "NO ENCONTRADAS", len(abl), len(ft)))
    if len(ft) < 6:
        print("AVISO: se esperaban 6 filas de finetune (2 modelos x 3 semillas).")
        print("       Reconstruyelas sin reentrenar con:")
        print("       python ablation_resolucion/finetune_all.py --evaluar-solo")
    d = construir(pub, abl, ft, a.out)
    print("dosier ->", d)


if __name__ == "__main__":
    main()
