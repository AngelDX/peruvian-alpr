#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Limpia el dataset Roboflow "Deteccion de Placas Peruanas" (v11) para poder usarlo
como dominio externo (fotos de celular/web) del detector movil.

Dos defectos del export original que este script corrige:

  1. Augmentacion horneada: 11 569 archivos que provienen de solo ~3 250 imagenes
     fuente. Se conserva UNA copia por fuente.
  2. Fuga entre particiones: el split se hizo por archivo DESPUES de aumentar, asi que
     la misma foto aparece en train y en test (73 % de las fuentes de test). Se
     reparticiona agrupando por imagen fuente.

Ademas remapea las clases (por defecto a una sola clase 'placa', ver --keep-classes),
valida las anotaciones y emite un reporte reproducible para la seccion Methods.

Uso:
    python clean_roboflow_dataset.py --src ../PlacasPeruanasv11 --out ../PlacasPeruanas_clean
"""
import argparse, json, os, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

# <nombre_fuente>.rf.<hash>.<ext>  ->  el grupo 1 identifica la imagen fuente
STEM_RE = re.compile(r"^(?P<stem>.+)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
SPLITS = ("train", "valid", "test")


def source_stem(filename):
    """Nombre de la imagen fuente, descartando el sufijo de variante de Roboflow."""
    m = STEM_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def read_label(path):
    """Lee un archivo de anotacion de formato MIXTO y lo normaliza.

    El export de Roboflow combina dos formatos en el mismo dataset:
      - 5 campos  -> caja YOLO clasica: cls xc yc w h
      - 1+2n campos (n>=3) -> poligono de segmentacion: cls x1 y1 x2 y2 ... xn yn

    Devuelve (objetos, errores, formatos). Cada objeto es un dict con la caja
    envolvente normalizada y, cuando existe, el poligono original (el cuadrilatero
    de la placa sirve para rectificar perspectiva antes del OCR).
    """
    objs, errors, formats = [], [], Counter()
    if not path.exists():
        return objs, ["label ausente"], formats
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"L{i}: campo no numerico")
            continue

        if len(vals) == 4:
            # Caja: cls xc yc w h
            xc, yc, w, h = vals
            poly = None
            formats["bbox"] += 1
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            # Poligono: al menos 3 vertices -> caja envolvente
            poly = [(min(max(vals[k], 0.0), 1.0), min(max(vals[k + 1], 0.0), 1.0))
                    for k in range(0, len(vals), 2)]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            formats["poligono"] += 1
        else:
            errors.append(f"L{i}: {len(parts)} campos, no es caja (5) ni poligono (1+2n, n>=3)")
            continue

        if w <= 0 or h <= 0:
            errors.append(f"L{i}: geometria degenerada (w={w:.4f} h={h:.4f})")
            continue
        if not all(-1e-6 <= v <= 1 + 1e-6 for v in (xc, yc, w, h)):
            errors.append(f"L{i}: coordenadas fuera de [0,1]")
            continue
        objs.append({
            "cls": cls,
            "box": (min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0), min(w, 1.0), min(h, 1.0)),
            "poly": poly,
        })
    return objs, errors, formats


def scan(src):
    """Indexa todos los archivos del export original, agrupados por imagen fuente."""
    groups = defaultdict(list)
    per_split = Counter()
    for split in SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            per_split[split] += 1
            groups[source_stem(img.name)].append(
                {"split": split, "img": img, "lbl": lbl_dir / (img.stem + ".txt")}
            )
    return groups, per_split


def leakage(groups):
    """Fuentes compartidas entre particiones en el export original."""
    by_split = {s: set() for s in SPLITS}
    for stem, variants in groups.items():
        for v in variants:
            by_split[v["split"]].add(stem)
    return {
        "train_valid": len(by_split["train"] & by_split["valid"]),
        "train_test": len(by_split["train"] & by_split["test"]),
        "valid_test": len(by_split["valid"] & by_split["test"]),
        "fuentes_test_tambien_en_train": len(by_split["test"] & by_split["train"]),
        "fuentes_test": len(by_split["test"]),
    }


def link_or_copy(src_path, dst_path):
    """Hardlink si se puede (mismo volumen, sin duplicar bytes); si no, copia."""
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
        return "link"
    except OSError:
        shutil.copy2(src_path, dst_path)
        return "copy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../PlacasPeruanasv11", help="export original de Roboflow")
    ap.add_argument("--out", default="../PlacasPeruanas_clean", help="dataset limpio de salida")
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,valid,test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-classes", action="store_true",
                    help="conserva placa_carro/placa_moto en vez de colapsar a 'placa'")
    ap.add_argument("--only", default=None,
                    help="lista separada por comas de clases a conservar, p.ej. 'placa_carro'. "
                         "Las anotaciones de las demas clases se descartan.")
    ap.add_argument("--neg-ratio", type=float, default=0.10,
                    help="con --only: fraccion maxima de imagenes sin objetos de interes que se "
                         "conservan como negativos explicitos (0 = descartarlas todas)")
    ap.add_argument("--task", choices=("detect", "segment"), default="detect",
                    help="detect: escribe cajas YOLO. segment: escribe poligonos "
                         "(las anotaciones que ya venian como caja se emiten como rectangulo)")
    ap.add_argument("--drop-empty", action="store_true",
                    help="descarta imagenes sin cajas en vez de conservarlas como negativos")
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    if not src.is_dir():
        sys.exit(f"No existe el dataset de origen: {src}")

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit("--ratios debe traer 3 valores que sumen 1.0")

    src_names = ["placa_carro", "placa_moto"]
    if args.only:
        keep = [c.strip() for c in args.only.split(",") if c.strip()]
        desconocidas = [c for c in keep if c not in src_names]
        if desconocidas:
            sys.exit(f"--only: clases desconocidas {desconocidas}; disponibles: {src_names}")
        out_names = keep
        remap = {src_names.index(c): i for i, c in enumerate(keep)}
    elif args.keep_classes:
        out_names, remap = src_names, {0: 0, 1: 1}
    else:
        out_names, remap = ["placa"], {0: 0, 1: 0}

    print(f"Leyendo {src} ...")
    groups, per_split_orig = scan(src)
    total_files = sum(per_split_orig.values())
    if not total_files:
        sys.exit("No se encontraron imagenes en el origen.")
    leak_before = leakage(groups)

    print(f"  {total_files} archivos -> {len(groups)} imagenes fuente unicas "
          f"({total_files/len(groups):.1f}x de inflacion)")
    print(f"  fuga original: {leak_before['fuentes_test_tambien_en_train']}"
          f"/{leak_before['fuentes_test']} fuentes de test tambien estaban en train")

    # --- Una variante por fuente. Eleccion determinista (orden de nombre) para
    # que el dataset limpio sea reproducible bit a bit.
    rng = random.Random(args.seed)
    candidatos, label_errors, empty_sources = {}, {}, []
    fmt_counts = Counter()
    for stem, variants in groups.items():
        pick = sorted(variants, key=lambda v: v["img"].name)[0]
        objs, errs, fmts = read_label(pick["lbl"])
        fmt_counts.update(fmts)
        if errs:
            label_errors[pick["img"].name] = errs
        if not objs:
            empty_sources.append(stem)
            if args.drop_empty:
                continue
        # Solo sobreviven los objetos de las clases pedidas.
        objs = [o for o in objs if o["cls"] in remap]
        candidatos[stem] = {"img": pick["img"], "objs": objs}

    # --- Negativos: imagenes que quedaron sin ningun objeto de interes. Conservar unas
    # pocas ayuda a suprimir falsos positivos (p.ej. ensenan a NO disparar sobre una placa
    # de moto); conservarlas todas desbalancea el entrenamiento hacia el fondo.
    positivos = [s for s, r in candidatos.items() if r["objs"]]
    negativos = [s for s, r in candidatos.items() if not r["objs"]]
    neg_max = int(round(len(positivos) * max(args.neg_ratio, 0.0)))
    negativos_orden = sorted(negativos)
    rng.shuffle(negativos_orden)
    negativos_keep = set(negativos_orden[:neg_max])
    chosen = {s: r for s, r in candidatos.items() if r["objs"] or s in negativos_keep}
    if negativos:
        print(f"  negativos: {len(negativos)} imagenes sin clases de interes -> "
              f"se conservan {len(negativos_keep)} ({args.neg_ratio:.0%} de los positivos)")

    # --- Estratos: se reparte cada combinacion de clases entre particiones para que
    # ninguna quede sub-representada (la clase minoritaria es la critica).
    strata = defaultdict(list)
    for stem, rec in chosen.items():
        classes = tuple(sorted({remap[o["cls"]] for o in rec["objs"]}))
        strata[classes if classes else ("neg",)].append(stem)
    assign = {}
    for key in sorted(strata, key=str):  # las claves mezclan tuplas de int y ("neg",)
        stems = sorted(strata[key])
        rng.shuffle(stems)
        n = len(stems)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        for s in stems[:n_tr]:
            assign[s] = "train"
        for s in stems[n_tr:n_tr + n_va]:
            assign[s] = "valid"
        for s in stems[n_tr + n_va:]:
            assign[s] = "test"

    # --- Escritura
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {s: {"imgs": 0, "boxes": Counter(), "empty": 0} for s in SPLITS}
    modes = Counter()
    for stem, rec in chosen.items():
        split = assign[stem]
        img_dst = out / split / "images" / rec["img"].name
        modes[link_or_copy(rec["img"], img_dst)] += 1
        lines = []
        for o in rec["objs"]:
            c = remap.get(o["cls"])
            if c is None:
                continue
            xc, yc, w, h = o["box"]
            if args.task == "segment":
                if o["poly"] is not None:
                    pts = o["poly"]
                else:  # venia como caja: se emite el rectangulo equivalente
                    x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                lines.append(f"{c} {coords}")
            else:
                lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats[split]["boxes"][c] += 1
        (out / split / "labels" / (rec["img"].stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[split]["imgs"] += 1
        if not lines:
            stats[split]["empty"] += 1

    (out / "data.yaml").write_text(
        "# Dataset limpio: deduplicado por imagen fuente y reparticionado sin fuga.\n"
        "# Generado por benchmark/clean_roboflow_dataset.py — no editar a mano.\n"
        f"path: {out.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(out_names)}\n"
        f"names: {out_names}\n\n"
        "# Fuente: Roboflow Universe, 'Deteccion de Placas Peruanas' v11 (CC BY 4.0)\n"
        "# https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9\n",
        encoding="utf-8")

    # La licencia CC BY 4.0 obliga a atribuir; queda junto a los datos derivados.
    (out / "ATTRIBUTION.md").write_text(
        "# Atribucion\n\n"
        "Obra derivada de **\"Deteccion de Placas Peruanas\" v11 (VERSION_4)**, publicado por el\n"
        "workspace `proyecto-de-tesis` en Roboflow Universe.\n\n"
        "- Fuente: https://universe.roboflow.com/proyecto-de-tesis/deteccion-de-placas-peruanas-ybiq9\n"
        "- Licencia: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/)\n\n"
        "## Modificaciones aplicadas\n\n"
        "El export original no es utilizable tal cual: la augmentacion viene horneada en los\n"
        "archivos y la particion se hizo por archivo despues de aumentar, de modo que la misma\n"
        "imagen fuente aparece simultaneamente en train y en test.\n\n"
        f"1. Deduplicacion por imagen fuente: {total_files} archivos -> {len(groups)} unicas.\n"
        f"2. Reparticion agrupada por fuente ({ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%}, "
        f"semilla {args.seed}), estratificada por clase, sin solape entre particiones.\n"
        "3. Normalizacion de anotaciones de formato mixto (cajas YOLO y poligonos de\n"
        "   segmentacion) a un unico formato.\n"
        f"4. Remapeo de clases a: {out_names}.\n\n"
        "Detalle cuantitativo en `clean_report.json`. Script: `benchmark/clean_roboflow_dataset.py`.\n",
        encoding="utf-8")

    # --- Verificacion: por construccion no puede haber fuga, pero se comprueba igual.
    out_groups, _ = scan(out)
    leak_after = leakage(out_groups)
    assert max(leak_after["train_valid"], leak_after["train_test"],
               leak_after["valid_test"]) == 0, "fuga residual tras reparticionar"

    dup_hist = Counter(len(v) for v in groups.values())
    report = {
        "origen": str(src),
        "salida": str(out),
        "semilla": args.seed,
        "ratios": ratios,
        "task": args.task,
        "clases_salida": out_names,
        "formatos_anotacion_origen": dict(fmt_counts),
        "original": {
            "archivos": total_files,
            "archivos_por_split": dict(per_split_orig),
            "fuentes_unicas": len(groups),
            "factor_inflacion": round(total_files / len(groups), 2),
            "fuga": leak_before,
            "copias_por_fuente": {str(k): v for k, v in sorted(dup_hist.items())},
        },
        "limpio": {
            "imagenes": sum(s["imgs"] for s in stats.values()),
            "por_split": {s: {"imgs": stats[s]["imgs"],
                              "cajas": {out_names[c]: n for c, n in sorted(stats[s]["boxes"].items())},
                              "sin_cajas": stats[s]["empty"]} for s in SPLITS},
            "fuga": leak_after,
        },
        "filtro_clases": {
            "only": args.only,
            "positivos": len(positivos),
            "sin_clases_de_interes": len(negativos),
            "negativos_conservados": len(negativos_keep),
            "neg_ratio": args.neg_ratio,
        },
        "anotaciones_con_error": label_errors,
        "fuentes_sin_cajas": len(empty_sources),
        "escritura": dict(modes),
    }
    (out / "clean_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    print(f"\nDataset limpio en {out}")
    for s in SPLITS:
        cajas = ", ".join(f"{out_names[c]}={n}" for c, n in sorted(stats[s]["boxes"].items()))
        print(f"  {s:5s}: {stats[s]['imgs']:5d} imgs  ({cajas})")
    if label_errors:
        print(f"  anotaciones con error: {len(label_errors)} (detalle en clean_report.json)")
    print(f"  fuga tras reparticionar: 0 fuentes compartidas [verificado]")
    print(f"  reporte: {out / 'clean_report.json'}")


if __name__ == "__main__":
    main()
