#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selecciona las placas del NUEVO gold limpio, garantizando cero solape con el entrenamiento.

MOTIVO. El gold actual solo aporta 126 placas no vistas en entrenamiento, y con ese tamano
la comparacion estudiante-vs-maestro tiene potencia 0.07: el resultado no significativo era
inevitable, hubiera o no diferencia real. El analisis de potencia (`gold2_power.py`) situa
en ~900 placas el tamano necesario para detectar 2 pp con potencia 0.80.

FUENTE. No hace falta capturar nada: la particion `test` del corpus aporta 2 992 placas
que, por construccion del split (agrupado por placa), NUNCA entraron en entrenamiento.
Solo falta verificarlas contra SUNARP.

Se excluyen ademas las placas ya presentes en cualquier conjunto etiquetado, para que el
esfuerzo de anotacion vaya integro a matriculas nuevas.

MUESTREO. Estratificado por dia/noche x cuartil de area de placa, proporcional al split de
test, y un unico recorte por placa. Asi el gold nuevo es comparable con el actual en
dificultad y no acumula casi-duplicados.

Salida: CSV compatible con `generate_active_sheets.py`, que genera las hojas Excel.

Uso:
    python build_gold2_selection.py --n 900
    python generate_active_sheets.py --sel gold2_selection.csv --out ../docs/verificacion_gold2 --batch 300
"""
import argparse, csv, gzip, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench_utils import norm_plate

CROP_RE = re.compile(r"^(?P<pref>.+?)_\(\s*(?P<h>\d+),\s*(?P<w>\d+),\s*3\)_(?P<plate>.+)\.jpg$")
# Cuartiles de area (px2) del corpus completo, de full_size_distribution.json
AREA_Q = (4095.0, 5136.0, 6510.0)


def placas_de_tsv(tsv):
    s = set()
    if not Path(tsv).exists():
        return s
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                p = norm_plate(row[1])
                if p:
                    s.add(p)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../Peruvian_ALPR_Dataset_final")
    ap.add_argument("--split", default="test",
                    help="particion(es) de origen separadas por coma; nunca 'train'")
    ap.add_argument("--dn", default="ambos",
                    help="clases luminicas a incluir, separadas por coma: ambos | dia | "
                         "crepusculo | noche. Usa `light_class` (elevacion solar real) si la "
                         "metadata enriquecida esta disponible; si no, la regla de reloj.")
    ap.add_argument("--excluir", default=None,
                    help="CSV de una seleccion previa cuyas placas hay que excluir")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="gold2_selection.csv")
    ap.add_argument("--flag-model", default="ckpt/lprnet_ft_2lotes.pt",
                    help="modelo para marcar discrepancias con la etiqueta silver ('' = omitir)")
    args = ap.parse_args()

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if "train" in splits:
        sys.exit("El gold no puede salir de 'train': ese es justamente el problema a evitar.")

    ds = Path(args.dataset).resolve()
    rng = random.Random(args.seed)

    # --- Placas ya etiquetadas en cualquier conjunto: no se vuelven a anotar.
    ya_etiquetadas = set()
    for tsv in ("data/rec_labels_gold.tsv", "data/rec_labels_gold_all.tsv",
                "data/rec_labels_ft_lote01.tsv"):
        ya_etiquetadas |= placas_de_tsv(tsv)
    entrenamiento = placas_de_tsv("data/rec_labels_train.tsv")
    print(f"Placas ya etiquetadas (excluidas): {len(ya_etiquetadas)}")
    print(f"Placas usadas en entrenamiento   : {len(entrenamiento)}")

    # --- Frecuencia de cada placa en el corpus. Importa mas de lo que parece: la fiabilidad
    # de la etiqueta silver crece con el numero de pasadas (una placa vista 50 veces recibe
    # una etiqueta por consenso; una vista una sola vez depende de una unica lectura de OCR).
    # Registrarla permite reponderar las metricas entre la vista por-placa y la por-evento.
    frecuencia = Counter()
    for tsv in ("data/rec_labels_train.tsv", "data/rec_labels_val.tsv",
                "data/rec_labels_test.tsv"):
        if Path(tsv).exists():
            with open(tsv, encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) >= 2:
                        p = norm_plate(row[1])
                        if p:
                            frecuencia[p] += 1

    # --- Placas de una seleccion previa, para no anotarlas dos veces
    ya_seleccionadas = set()
    if args.excluir and Path(args.excluir).exists():
        with open(args.excluir, encoding="utf-8") as f:
            ya_seleccionadas = {r["silver"] for r in csv.DictReader(f)}
        print(f"Placas de {args.excluir} (excluidas): {len(ya_seleccionadas)}")

    # --- Metadata. Se prefiere la enriquecida, que trae `light_class` calculada con la
    # elevacion solar real en vez del corte horario fijo (ver recompute_daynight.py).
    meta_path = ds / "metadata_full_light.csv.gz"
    if not meta_path.exists():
        meta_path = ds / "metadata_full.csv.gz"
        print("AVISO: sin metadata enriquecida; se usa la regla de reloj day_night")
    else:
        print(f"Clasificacion luminica: light_class (elevacion solar real)")

    clases = [c.strip() for c in args.dn.split(",") if c.strip()]
    meta = {}
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits:
                continue
            pref = row["filename"].split("_(")[0]
            luz = row.get("light_class") or row["day_night"]
            meta[pref] = (norm_plate(row["plate"]), row["month"], luz, row["split"])
    print(f"Frames en {splits}: {len(meta)}")

    # --- Recortes disponibles, agrupados por placa
    por_placa = defaultdict(list)
    for sp in splits:
        for e in os.scandir(ds / "crops" / sp):
            m = CROP_RE.match(e.name)
            if not m:
                continue
            info = meta.get(m.group("pref"))
            if info is None:
                continue
            placa, mes, dn, split_origen = info
            if not placa:
                continue
            if "ambos" not in clases and dn not in clases:
                continue
            por_placa[placa].append({
                "path": e.path, "fn": e.name, "silver": placa, "mes": mes, "day_night": dn,
                "split": split_origen, "area": int(m.group("h")) * int(m.group("w")),
            })

    # --- Filtro de seguridad: fuera todo lo visto en entrenamiento o ya etiquetado.
    descartadas_train = {p for p in por_placa if p in entrenamiento}
    if descartadas_train:
        print(f"AVISO: {len(descartadas_train)} placas de '{args.split}' aparecen en train; se excluyen")
    candidatas = {p: v for p, v in por_placa.items()
                  if p not in entrenamiento and p not in ya_etiquetadas
                  and p not in ya_seleccionadas}
    print(f"Placas candidatas (nunca vistas, sin etiquetar): {len(candidatas)}")
    if len(candidatas) < args.n:
        print(f"AVISO: solo hay {len(candidatas)} candidatas para las {args.n} pedidas")

    # --- Un recorte por placa
    elegidos = {p: rng.choice(sorted(v, key=lambda d: d["fn"])) for p, v in candidatas.items()}

    # --- Estratificacion proporcional por dia/noche x cuartil de area
    def estrato(d):
        return (d["day_night"], sum(d["area"] >= q for q in AREA_Q))

    grupos = defaultdict(list)
    for d in elegidos.values():
        grupos[estrato(d)].append(d)
    total = sum(len(v) for v in grupos.values())
    n_obj = min(args.n, total)

    cuota, restos = {}, []
    for k in sorted(grupos, key=str):
        exacto = n_obj * len(grupos[k]) / total
        cuota[k] = int(exacto)
        restos.append((exacto - int(exacto), k))
    restos.sort(reverse=True, key=lambda t: (t[0], str(t[1])))
    for _, k in restos[: n_obj - sum(cuota.values())]:
        cuota[k] += 1

    seleccion = []
    for k in sorted(grupos, key=str):
        pool = sorted(grupos[k], key=lambda d: d["fn"])
        rng.shuffle(pool)
        seleccion.extend(pool[: cuota[k]])
    rng.shuffle(seleccion)
    print(f"Seleccionadas: {len(seleccion)} placas")

    # --- Marcado de discrepancias: ayuda al anotador a priorizar los casos dudosos.
    motivos = {d["fn"]: "" for d in seleccion}
    if args.flag_model and Path(args.flag_model).exists():
        from paired_stats import predecir_ckpt
        print(f"  contrastando con {args.flag_model} para marcar discrepancias ...")
        preds = predecir_ckpt(args.flag_model, [d["path"] for d in seleccion])
        n_disc = 0
        for d, pred in zip(seleccion, preds):
            if norm_plate(pred) != d["silver"]:
                motivos[d["fn"]] = "modelos discrepan"
                n_disc += 1
        print(f"  {n_disc} discrepancias marcadas ({100*n_disc/len(seleccion):.1f} %)")

    # --- Salida
    destino = Path(args.out).resolve()
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "fn", "silver", "motivo", "mes", "day_night", "area_px2",
                    "frames_de_la_placa", "split_origen"])
        for d in seleccion:
            w.writerow([d["path"], d["fn"], d["silver"], motivos[d["fn"]],
                        d["mes"], d["day_night"], d["area"], frecuencia.get(d["silver"], 0),
                        d.get("split", "")])

    dn = Counter(d["day_night"] for d in seleccion)
    areas = sorted(d["area"] for d in seleccion)
    print("\nComposicion luminica: " + "  ".join(f"{k}={v}" for k, v in sorted(dn.items())))
    print(f"Area px2: q1={areas[len(areas)//4]}  mediana={areas[len(areas)//2]}  "
          f"q3={areas[3*len(areas)//4]}  (corpus: 4095 / 5136 / 6510)")
    print(f"Meses cubiertos: {len(set(d['mes'] for d in seleccion))}")
    perfil = Counter()
    for d in seleccion:
        n = frecuencia.get(d["silver"], 0)
        perfil["1-2" if n <= 2 else "3-9" if n < 10 else "10-49" if n < 50 else "50+"] += 1
    tot = sum(perfil.values())
    print("Frecuencia de la placa en el corpus: " +
          "  ".join(f"{k}={100*perfil[k]/tot:.1f}%" for k in ("1-2", "3-9", "10-49", "50+")))
    print(f"\nCSV: {destino}")
    print("Siguiente paso:\n  python generate_active_sheets.py --sel "
          f"{destino.name} --out ../docs/verificacion_gold2 --batch 300")


if __name__ == "__main__":
    main()
