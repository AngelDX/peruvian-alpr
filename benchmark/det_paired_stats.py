#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrastes de deteccion, distinguiendo el diseno apareado del no apareado.

Dos comparaciones que parecen del mismo tipo pero NO lo son:

  (a) APAREADA -- dos detectores sobre EL MISMO test (p. ej. yolov8s vs yolo11n_mobile
      sobre test_movil). Cada imagen aporta un par correlacionado -> McNemar exacta.

  (b) NO APAREADA -- el MISMO detector sobre DOS test distintos (test_camara vs test_movil).
      No hay correspondencia entre imagenes: son conjuntos distintos, de distinto tamano y
      distinta dificultad intrinseca. Aqui McNemar no aplica; se usa un contraste de dos
      proporciones independientes (Newcombe para el IC de la diferencia). Ademas la
      diferencia NO es atribuible solo al dominio: los conjuntos difieren tambien en
      anotador (silver vs humano) y en convencion de etiquetado. Es una limitacion del
      diseno, no algo que el estadistico pueda resolver.

Metrica por imagen: exito = todas las cajas de verdad-terreno quedan emparejadas con alguna
prediccion a IoU >= umbral. Es binaria e interpretable ("encontro todas las placas").

Uso:
    python det_paired_stats.py --a det_runs/yolov8s/weights/best.pt \
        --b "D:/.../yolo11n_mobile/weights/best.pt" --out ../docs/paper_materials/09_estadistica_deteccion.json
"""
import argparse, json, math
from pathlib import Path

import numpy as np


def cargar_gt(labels_dir, img_paths):
    """Cajas de verdad-terreno por imagen, en xyxy normalizado."""
    gt = []
    for p in img_paths:
        f = Path(labels_dir) / (Path(p).stem + ".txt")
        cajas = []
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                v = line.split()
                if len(v) == 5:
                    _, xc, yc, w, h = (float(x) for x in v)
                    cajas.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
        gt.append(np.array(cajas, dtype=np.float32).reshape(-1, 4))
    return gt


def iou_matriz(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def exito_por_imagen(modelo_path, img_paths, gt, iou_thr, conf):
    """Vector booleano: la imagen i tiene TODAS sus placas detectadas."""
    from ultralytics import YOLO
    modelo = YOLO(str(modelo_path))
    ok = np.zeros(len(img_paths), dtype=bool)
    B = 32
    for i in range(0, len(img_paths), B):
        lote = img_paths[i:i + B]
        res = modelo.predict(lote, conf=conf, verbose=False, imgsz=640)
        for j, r in enumerate(res):
            k = i + j
            verdad = gt[k]
            if len(verdad) == 0:
                ok[k] = True   # sin placas que encontrar
                continue
            pred = r.boxes.xyxyn.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            m = iou_matriz(verdad, pred.astype(np.float32))
            ok[k] = bool(len(pred) and (m.max(axis=1) >= iou_thr).all())
    return ok


def mcnemar_exacto(a, b):
    b10 = int(np.sum(a & ~b)); b01 = int(np.sum(~a & b))
    n = b10 + b01
    if n == 0:
        return {"b10": 0, "b01": 0, "n_discordantes": 0, "p": 1.0}
    k = min(b10, b01)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b10": b10, "b01": b01, "n_discordantes": n, "p": min(1.0, 2 * cola)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def newcombe(k1, n1, k2, n2):
    """IC 95 % de la diferencia de dos proporciones INDEPENDIENTES (Newcombe, metodo 10).

    Es el contraste adecuado cuando no hay apareamiento: combina los IC de Wilson de cada
    proporcion en vez de asumir normalidad de la diferencia.
    """
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="detector A (base, entrenado solo con la camara)")
    ap.add_argument("--b", required=True, help="detector B (movil, dataset fusionado)")
    ap.add_argument("--dataset", default="../MobileALPR_dataset")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="../docs/paper_materials/09_estadistica_deteccion.json")
    args = ap.parse_args()

    ds = Path(args.dataset).resolve()
    reporte = {"iou": args.iou, "conf": args.conf,
               "metrica": "imagen con TODAS sus placas detectadas (IoU >= umbral)"}

    conjuntos = {}
    for nombre in ("test_movil", "test_camara"):
        imgs = sorted(str(p) for p in (ds / nombre / "images").iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        conjuntos[nombre] = (imgs, cargar_gt(ds / nombre / "labels", imgs))
        print(f"{nombre}: {len(imgs)} imagenes")

    # ---------- (a) COMPARACION APAREADA: dos detectores, mismo test movil ----------
    imgs, gt = conjuntos["test_movil"]
    print("\nInfiriendo detector A sobre test_movil ...")
    ok_a = exito_por_imagen(args.a, imgs, gt, args.iou, args.conf)
    print("Infiriendo detector B sobre test_movil ...")
    ok_b = exito_por_imagen(args.b, imgs, gt, args.iou, args.conf)

    n = len(imgs)
    mc = mcnemar_exacto(ok_b, ok_a)   # B vs A: se espera que B (movil) gane
    la, ua = wilson(int(ok_a.sum()), n)
    lb, ub = wilson(int(ok_b.sum()), n)
    print(f"\n--- APAREADA (mismo test_movil, n={n})")
    print(f"  A (camara): {ok_a.sum():4d}/{n}  = {ok_a.mean():.4f}  IC[{la:.4f},{ua:.4f}]")
    print(f"  B (movil) : {ok_b.sum():4d}/{n}  = {ok_b.mean():.4f}  IC[{lb:.4f},{ub:.4f}]")
    print(f"  McNemar exacta: b10={mc['b10']} b01={mc['b01']}  p={mc['p']:.3e}")
    reporte["apareada_mismo_test_movil"] = {
        "n": n,
        "A": {"modelo": str(args.a), "exitos": int(ok_a.sum()), "tasa": float(ok_a.mean()),
              "ic95_wilson": [la, ua]},
        "B": {"modelo": str(args.b), "exitos": int(ok_b.sum()), "tasa": float(ok_b.mean()),
              "ic95_wilson": [lb, ub]},
        "mcnemar_B_vs_A": mc,
        "prueba": "McNemar exacta (las mismas imagenes evaluadas por los dos modelos)",
    }

    # ---------- (b) COMPARACION NO APAREADA: mismo detector, dos test distintos ----------
    imgs_c, gt_c = conjuntos["test_camara"]
    print("\nInfiriendo detector A sobre test_camara ...")
    ok_a_cam = exito_por_imagen(args.a, imgs_c, gt_c, args.iou, args.conf)

    k1, n1 = int(ok_a_cam.sum()), len(imgs_c)
    k2, n2 = int(ok_a.sum()), n
    lo, hi = newcombe(k1, n1, k2, n2)
    print(f"\n--- NO APAREADA (test_camara n={n1} vs test_movil n={n2}, detector A)")
    print(f"  camara: {k1}/{n1} = {k1/n1:.4f}")
    print(f"  movil : {k2}/{n2} = {k2/n2:.4f}")
    print(f"  diferencia = {k1/n1 - k2/n2:+.4f}  IC 95 % Newcombe [{lo:+.4f}, {hi:+.4f}]")
    reporte["no_apareada_entre_dominios"] = {
        "modelo": str(args.a),
        "camara": {"exitos": k1, "n": n1, "tasa": k1 / n1},
        "movil": {"exitos": k2, "n": n2, "tasa": k2 / n2},
        "diferencia": k1 / n1 - k2 / n2,
        "ic95_newcombe": [lo, hi],
        "prueba": "dos proporciones independientes (conjuntos distintos, sin apareamiento)",
        "advertencia": ("Los dos conjuntos difieren ademas en procedencia de las anotaciones "
                        "(silver vs humanas) y en convencion de etiquetado, de modo que la "
                        "diferencia no es atribuible unicamente al cambio de dominio."),
    }

    destino = Path(args.out).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {destino}")


if __name__ == "__main__":
    main()
