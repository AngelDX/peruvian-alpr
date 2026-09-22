#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye el dataset de entrenamiento del detector MOVIL de placas de auto.

Fusiona dos dominios:
  A) camara fija de la universidad (Peruvian_ALPR_Dataset_final, 579 292 frames)
  B) fotos de celular/web (PlacasPeruanas_clean, derivado de Roboflow CC BY 4.0)

Tres decisiones que condicionan el resultado:

  1. Se SUBMUESTREA el dominio A. Volcarlo entero (579k contra ~1.5k) haria que el
     detector reaprendiera el sesgo de la camara fija y no generalizara a celulares,
     que es justo lo que se quiere evitar.
  2. Un frame por placa. El dominio A tiene decenas de frames casi identicos del mismo
     vehiculo (mismo coche a un segundo de distancia); muestrear al azar llenaria el
     conjunto de casi-duplicados en vez de diversidad.
  3. DOS conjuntos de test separados (test_camara / test_movil). Promediarlos en una
     sola cifra esconderia exactamente la brecha de dominio que interesa medir.

El muestreo del dominio A respeta la particion original (train/val/test), que ya
garantiza que ninguna placa cruza entre particiones, y se estratifica por
mes x dia/noche x cuartil de area de placa.

Uso:
    python build_mobile_dataset.py --out ../MobileALPR_dataset
"""
import argparse, csv, gzip, json, os, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

# Cuartiles de area de placa (px2) del corpus completo, de full_size_distribution.json.
AREA_Q = (4095.0, 5136.0, 6510.0)
CAM_SPLIT_OUT = {"train": "train", "val": "valid", "test": "test_camara"}
MOB_SPLIT_OUT = {"train": "train", "valid": "valid", "test": "test_movil"}
OUT_SPLITS = ("train", "valid", "test_camara", "test_movil")


def area_bin(area):
    return sum(area >= q for q in AREA_Q)


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def plate_area_px2(label_path, img_w=1280, img_h=720):
    """Area de la caja mas grande del label, en pixeles."""
    best = 0.0
    try:
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) == 5:
                best = max(best, float(p[3]) * img_w * float(p[4]) * img_h)
    except OSError:
        return 0.0
    return best


def load_camera_metadata(meta_path):
    """filename -> (split, month, day_night, plate), agrupado por particion."""
    by_split = defaultdict(list)
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            by_split[row["split"]].append(
                (row["filename"], row["month"], row["day_night"], row["plate"])
            )
    return by_split


def sample_camera(rows, n_target, cam_root, split, rng, report):
    """Muestreo estratificado por mes x dia/noche x cuartil de area, un frame por placa."""
    if n_target >= len(rows):
        print(f"    aviso: se pidieron {n_target} frames y solo hay {len(rows)}")

    # --- Un frame por placa (elegido al azar entre los suyos).
    por_placa = defaultdict(list)
    for fn, month, dn, plate in rows:
        por_placa[plate].append((fn, month, dn))
    candidatos = []
    for plate in sorted(por_placa):
        fn, month, dn = rng.choice(sorted(por_placa[plate]))
        candidatos.append((fn, month, dn, plate))
    report["placas_unicas"] = len(por_placa)

    # --- Estratos mes x dia/noche, con reparto proporcional.
    estratos = defaultdict(list)
    for c in candidatos:
        estratos[(c[1], c[2])] += [c]
    total = len(candidatos)
    n_target = min(n_target, total)

    cuota, resto = {}, []
    for k in sorted(estratos):
        exacto = n_target * len(estratos[k]) / total
        cuota[k] = int(exacto)
        resto.append((exacto - int(exacto), k))
    # El remanente por redondeo va a los estratos con mayor parte fraccionaria.
    resto.sort(reverse=True)
    for _, k in resto[: n_target - sum(cuota.values())]:
        cuota[k] += 1

    lbl_dir = cam_root / "labels" / split
    elegidos = []
    for k in sorted(estratos):
        need = cuota[k]
        if need <= 0:
            continue
        pool = sorted(estratos[k])
        rng.shuffle(pool)
        pool = pool[: max(need * 4, need)]  # solo se leen labels del pool, no de todo
        por_bin = defaultdict(list)
        for fn, month, dn, plate in pool:
            a = plate_area_px2(lbl_dir / (Path(fn).stem + ".txt"))
            por_bin[area_bin(a)].append((fn, a))
        # Round-robin entre cuartiles de tamano para no sesgar hacia placas grandes.
        orden = sorted(por_bin)
        i = 0
        while need > 0 and any(por_bin[b] for b in orden):
            b = orden[i % len(orden)]
            if por_bin[b]:
                elegidos.append(por_bin[b].pop())
                need -= 1
            i += 1
    return elegidos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-root", default="../Peruvian_ALPR_Dataset_final")
    ap.add_argument("--mobile-root", default="../PlacasPeruanas_clean")
    ap.add_argument("--out", default="../MobileALPR_dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-images", action="store_true",
                    help="usa images/ en vez del espejo anonimizado images_anonymized/")
    args = ap.parse_args()

    cam_root = Path(args.camera_root).resolve()
    mob_root = Path(args.mobile_root).resolve()
    out = Path(args.out).resolve()
    for p, nombre in ((cam_root, "camera-root"), (mob_root, "mobile-root")):
        if not p.is_dir():
            sys.exit(f"No existe {nombre}: {p}")

    img_sub = "images" if args.raw_images else "images_anonymized"
    if not (cam_root / img_sub).is_dir():
        sys.exit(f"No existe {cam_root / img_sub}")
    if not args.raw_images:
        print(f"Dominio camara: usando el espejo anonimizado ({img_sub}/)")

    rng = random.Random(args.seed)
    report = {"semilla": args.seed, "camara": {}, "movil": {}, "salida": str(out)}

    print(f"Leyendo metadata de {cam_root.name} ...")
    by_split = load_camera_metadata(cam_root / "metadata_full.csv.gz")
    print("  frames por particion: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(by_split.items())))

    if out.exists():
        shutil.rmtree(out)
    for s in OUT_SPLITS:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)

    modes = Counter()
    stats = {s: Counter() for s in OUT_SPLITS}
    areas = defaultdict(list)

    # --- Dominio A: camara fija (submuestreado)
    objetivos = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, n_target in objetivos.items():
        dst_split = CAM_SPLIT_OUT[split]
        rep = {}
        print(f"  muestreando camara/{split}: objetivo {n_target} ...")
        elegidos = sample_camera(by_split[split], n_target, cam_root, split, rng, rep)
        for fn, a in elegidos:
            stem = Path(fn).stem
            src_img = cam_root / img_sub / split / fn
            src_lbl = cam_root / "labels" / split / (stem + ".txt")
            if not src_img.exists() or not src_lbl.exists():
                stats[dst_split]["faltantes"] += 1
                continue
            modes[link_or_copy(src_img, out / dst_split / "images" / ("cam_" + fn))] += 1
            shutil.copy2(src_lbl, out / dst_split / "labels" / ("cam_" + stem + ".txt"))
            stats[dst_split]["camara"] += 1
            areas[dst_split].append(a)
        rep.update({"objetivo": n_target, "obtenidos": stats[dst_split]["camara"]})
        report["camara"][split] = rep

    # --- Dominio B: celular/web (integro)
    for split, dst_split in MOB_SPLIT_OUT.items():
        src_img_dir = mob_root / split / "images"
        if not src_img_dir.is_dir():
            continue
        n = 0
        for img in sorted(src_img_dir.iterdir()):
            lbl = mob_root / split / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            modes[link_or_copy(img, out / dst_split / "images" / ("mob_" + img.name))] += 1
            shutil.copy2(lbl, out / dst_split / "labels" / ("mob_" + img.stem + ".txt"))
            stats[dst_split]["movil"] += 1
            n += 1
        report["movil"][split] = {"imagenes": n, "destino": dst_split}

    # --- data.yaml + yamls de evaluacion por dominio
    def yaml(test_dir):
        return (f"path: {out.as_posix()}\n"
                "train: train/images\n"
                "val: valid/images\n"
                f"test: {test_dir}/images\n\n"
                "nc: 1\n"
                "names: ['placa']\n")

    (out / "data.yaml").write_text(
        "# Detector movil de placas de auto. Generado por benchmark/build_mobile_dataset.py\n"
        "# test = dominio movil (el objetivo). Para el otro dominio: data_test_camara.yaml\n"
        + yaml("test_movil"), encoding="utf-8")
    (out / "data_test_camara.yaml").write_text(
        "# Evaluacion IN-DOMAIN (camara fija):  yolo val data=data_test_camara.yaml split=test\n"
        + yaml("test_camara"), encoding="utf-8")
    (out / "data_test_movil.yaml").write_text(
        "# Evaluacion CROSS-DOMAIN (celular/web):  yolo val data=data_test_movil.yaml split=test\n"
        + yaml("test_movil"), encoding="utf-8")

    # --- Representatividad del submuestreo: los cuartiles de area deben parecerse
    # a los del corpus completo (q1=4095, mediana=5136, q3=6510).
    for k, v in areas.items():
        if v:
            v.sort()
            report.setdefault("area_px2_muestreada", {})[k] = {
                "n": len(v),
                "q1": round(v[len(v) // 4], 1),
                "mediana": round(v[len(v) // 2], 1),
                "q3": round(v[3 * len(v) // 4], 1),
            }
    report["corpus_area_px2_referencia"] = {"q1": AREA_Q[0], "mediana": AREA_Q[1], "q3": AREA_Q[2]}
    report["por_split"] = {s: dict(stats[s]) for s in OUT_SPLITS}
    report["escritura"] = dict(modes)
    report["imagenes_anonimizadas"] = not args.raw_images
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    print(f"\nDataset en {out}")
    for s in OUT_SPLITS:
        c, m = stats[s]["camara"], stats[s]["movil"]
        print(f"  {s:12s}: {c + m:5d} imgs  (camara={c}, movil={m})")
    if any(stats[s]["faltantes"] for s in OUT_SPLITS):
        print("  aviso: hubo frames sin imagen o sin label (ver build_report.json)")
    print(f"  reporte: {out / 'build_report.json'}")


if __name__ == "__main__":
    main()
