#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reparticiona el dataset de DETECCION sin fuga, para la revision 1 de Sensors.

El Revisor 2 senala que las 2 600 imagenes anotadas se repartieron por archivo y
al azar, de modo que el 76 % comparte cadena de matricula con otra particion y el
54 % comparte rafaga temporal. Las metricas de deteccion que reporta el articulo
son, por tanto, optimistas. Este script rehace el reparto agrupando por entidad,
igual que ya se hace en reconocimiento, y deja medida la fuga antes y despues.

Criterio de agrupamiento (union-find sobre las imagenes):
  a) misma cadena de matricula, cuando el nombre de archivo trae una lectura con
     formato nacional LNN-DDD;
  b) marcas temporales separadas por menos de VENTANA segundos, que es lo que
     define una misma rafaga del mismo vehiculo.
Las dos condiciones se unen, de modo que ningun grupo queda partido por el corte.
Un grupo entero cae siempre del mismo lado.

Salida: det_split_nolk/{train,val,test}.txt + data.yaml (listas de rutas, sin copiar
imagenes) y un informe JSON con la auditoria de fuga de ambos repartos.

Uso (desde benchmark/):
    python resplit_det_nolk.py                    # reparto 70/15/15, semilla 0
    python resplit_det_nolk.py --solo-auditoria   # solo mide la fuga actual
"""
import os
import re
import csv
import json
import glob
import random
import shutil
import argparse
import collections

BENCH = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BENCH, "det_data")
DST = os.path.join(BENCH, "det_split_nolk")
SPLITS = ("train", "val", "test")

# 1741362991_182673_0_73_-720-1280-3-_V5B-631_jpg.rf.<hash>.jpg
PAT = re.compile(r"^(\d{9,10})(?:[._]\d+)?.*?_([A-Za-z0-9-]+)_+jpg\.rf\.[0-9a-f]+\.jpg$")
PLACA = re.compile(r"^[A-Z0-9]{3}-[A-Z0-9]{3}$")


# --------------------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def leer_imagenes(raiz):
    """Devuelve [(ruta, split_original, timestamp, placa_o_None)] ordenado por ts."""
    filas = []
    for sp in SPLITS:
        for f in sorted(glob.glob(os.path.join(raiz, "images", sp, "*.jpg"))):
            b = os.path.basename(f)
            m = PAT.match(b)
            if m:
                ts = int(m.group(1))
                tok = m.group(2).upper()
                placa = tok if PLACA.match(tok) else None
            else:
                # Un solo archivo del corpus no encaja en el patron; se trata como
                # grupo propio en vez de descartarlo.
                ts, placa = 0, None
            filas.append((f, sp, ts, placa))
    filas.sort(key=lambda r: r[2])
    return filas


def agrupar(filas, ventana):
    uf = UF(len(filas))
    por_placa = collections.defaultdict(list)
    for i, (_, _, _, placa) in enumerate(filas):
        if placa:
            por_placa[placa].append(i)
    for idxs in por_placa.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    # rafagas: la lista ya viene ordenada por marca temporal
    for i in range(1, len(filas)):
        if filas[i][2] and filas[i - 1][2] and filas[i][2] - filas[i - 1][2] <= ventana:
            uf.union(i - 1, i)
    grupos = collections.defaultdict(list)
    for i in range(len(filas)):
        grupos[uf.find(i)].append(i)
    return list(grupos.values())


def auditar(filas, asignacion, ventana):
    """% de imagenes que comparten placa o rafaga con otra particion."""
    n = len(filas)
    por_placa = collections.defaultdict(set)
    for i, (_, _, _, placa) in enumerate(filas):
        if placa:
            por_placa[placa].add(asignacion[i])
    comp_placa = 0
    for i, (_, _, _, placa) in enumerate(filas):
        if placa and len(por_placa[placa]) > 1:
            comp_placa += 1
    # rafagas
    rafaga, actual = [0] * n, 0
    for i in range(1, n):
        if not (filas[i][2] and filas[i - 1][2] and filas[i][2] - filas[i - 1][2] <= ventana):
            actual += 1
        rafaga[i] = actual
    por_rafaga = collections.defaultdict(set)
    for i in range(n):
        por_rafaga[rafaga[i]].add(asignacion[i])
    comp_raf = sum(1 for i in range(n) if len(por_rafaga[rafaga[i]]) > 1)
    return {"img_comparte_placa_pct": round(100.0 * comp_placa / n, 2),
            "img_comparte_rafaga_pct": round(100.0 * comp_raf / n, 2),
            "n_rafagas": actual + 1}


def repartir(grupos, filas, objetivo, semilla):
    """Asigna grupos completos hasta aproximar la proporcion pedida por imagenes."""
    rng = random.Random(semilla)
    g = sorted(grupos, key=len, reverse=True)
    # Los grupos grandes primero, al split que mas lejos este de su cuota: es el
    # reparto voraz habitual y deja desviaciones de pocas imagenes.
    total = len(filas)
    cuota = {s: objetivo[s] * total for s in SPLITS}
    acum = {s: 0 for s in SPLITS}
    asignacion = [None] * total
    for grupo in g:
        falta = {s: cuota[s] - acum[s] for s in SPLITS}
        sp = max(SPLITS, key=lambda s: (falta[s], rng.random()))
        for i in grupo:
            asignacion[i] = sp
        acum[sp] += len(grupo)
    return asignacion, acum


def materializar(filas, asignacion, src, dst):
    """Escribe el reparto como listas de rutas, sin duplicar las imagenes.

    Ultralytics admite que train/val/test apunten a un .txt con una ruta de imagen
    por linea, y deriva la etiqueta sustituyendo /images/ por /labels/. Como el
    dataset original conserva esa estructura, basta con listar: no se copia nada,
    el corpus anotado sigue siendo el mismo y el reparto queda en tres ficheros de
    texto que caben en el repositorio.
    """
    os.makedirs(dst, exist_ok=True)
    cuenta = {}
    for sp in SPLITS:
        rutas = [os.path.abspath(filas[i][0]) for i in range(len(filas)) if asignacion[i] == sp]
        with open(os.path.join(dst, "%s.txt" % sp), "w", encoding="utf-8") as fh:
            fh.write("\n".join(rutas) + "\n")
        cuenta[sp] = len(rutas)
    with open(os.path.join(dst, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# Reparto sin fuga: agrupado por matricula y por rafaga temporal.\n"
                 "# Las imagenes son las mismas de det_data; aqui solo cambia el reparto.\n"
                 "path: %s\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
                 "nc: 1\nnames: ['plate']\n" % os.path.abspath(dst).replace("\\", "/"))
    return cuenta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--ventana", type=int, default=5,
                    help="segundos que definen una misma rafaga (por defecto 5)")
    ap.add_argument("--props", default="0.70,0.15,0.15")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--solo-auditoria", action="store_true")
    ap.add_argument("--informe", default=os.path.join(BENCH, "results", "det_resplit_report.json"))
    a = ap.parse_args()

    filas = leer_imagenes(a.src)
    print("imagenes: %d" % len(filas), flush=True)
    con_placa = sum(1 for r in filas if r[3])
    print("con lectura de matricula en el nombre: %d (%.1f %%)"
          % (con_placa, 100.0 * con_placa / len(filas)), flush=True)

    orig = [r[1] for r in filas]
    aud_orig = auditar(filas, orig, a.ventana)
    print("\nreparto ORIGINAL (por archivo, al azar)", flush=True)
    print("  %s" % collections.Counter(orig), flush=True)
    print("  comparte placa con otra particion : %5.2f %%" % aud_orig["img_comparte_placa_pct"], flush=True)
    print("  comparte rafaga con otra particion: %5.2f %%" % aud_orig["img_comparte_rafaga_pct"], flush=True)
    print("  rafagas detectadas (ventana %ds)  : %d" % (a.ventana, aud_orig["n_rafagas"]), flush=True)

    grupos = agrupar(filas, a.ventana)
    print("\ngrupos tras unir placa y rafaga: %d (mayor: %d imagenes)"
          % (len(grupos), max(len(g) for g in grupos)), flush=True)

    p = [float(x) for x in a.props.split(",")]
    objetivo = dict(zip(SPLITS, p))
    asignacion, acum = repartir(grupos, filas, objetivo, a.semilla)
    aud_new = auditar(filas, asignacion, a.ventana)
    print("\nreparto NUEVO (agrupado por entidad)", flush=True)
    print("  %s" % acum, flush=True)
    print("  %% real: " + ", ".join("%s %.1f" % (s, 100.0 * acum[s] / len(filas)) for s in SPLITS), flush=True)
    print("  comparte placa con otra particion : %5.2f %%" % aud_new["img_comparte_placa_pct"], flush=True)
    print("  comparte rafaga con otra particion: %5.2f %%" % aud_new["img_comparte_rafaga_pct"], flush=True)

    os.makedirs(os.path.dirname(a.informe), exist_ok=True)
    with open(a.informe, "w", encoding="utf-8") as fh:
        json.dump({"n_imagenes": len(filas), "ventana_s": a.ventana, "semilla": a.semilla,
                   "grupos": len(grupos), "original": {"conteo": dict(collections.Counter(orig)),
                                                       "fuga": aud_orig},
                   "nuevo": {"conteo": acum, "fuga": aud_new}}, fh, indent=2, ensure_ascii=False)
    print("\ninforme -> %s" % a.informe, flush=True)

    if a.solo_auditoria:
        print("(--solo-auditoria: no se escribe el dataset)", flush=True)
        return
    cuenta = materializar(filas, asignacion, a.src, a.dst)
    print("listas    -> %s  (%s)" % (a.dst, cuenta), flush=True)
    print("\nSiguiente paso:", flush=True)
    print("  python det_benchmark.py --data det_split_nolk/data.yaml --epochs 60 "
          "--out det_results_nolk", flush=True)


if __name__ == "__main__":
    main()
