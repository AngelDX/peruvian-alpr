#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evalua un detector en los DOS dominios por separado y reporta la brecha.

in-domain   = camara fija de la universidad (test_camara)
cross-domain = fotos de celular/web         (test_movil)

La cifra que interesa no es ninguna de las dos por separado sino su diferencia:
cuanto se degrada un detector entrenado con una sola camara al cambiar de optica.

Uso:
    python crossdomain_eval.py --model det_runs/yolov8s/weights/best.pt --name yolov8s_camara
    python crossdomain_eval.py --model det_runs/yolov8s/weights/best.pt --name base --json out.json
"""
import argparse, json, sys
from pathlib import Path

DOMINIOS = [
    ("in_domain_camara", "data_test_camara.yaml"),
    ("cross_domain_movil", "data_test_movil.yaml"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ruta al .pt")
    ap.add_argument("--dataset", default="../MobileALPR_dataset")
    ap.add_argument("--name", default=None, help="etiqueta para el reporte")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--json", default=None, help="ruta del reporte (por defecto junto al dataset)")
    args = ap.parse_args()

    from ultralytics import YOLO

    ds = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        sys.exit(f"No existe el modelo: {model_path}")
    nombre = args.name or model_path.parent.parent.name

    modelo = YOLO(str(model_path))
    print(f"Modelo: {model_path}")
    print(f"Clases: {modelo.names}\n")

    resultados = {}
    for clave, yaml_name in DOMINIOS:
        yml = ds / yaml_name
        if not yml.exists():
            sys.exit(f"Falta {yml}; genera el dataset con build_mobile_dataset.py")
        print(f"--- {clave} ({yaml_name})")
        m = modelo.val(data=str(yml), split="test", imgsz=args.imgsz,
                       verbose=False, plots=False).box
        resultados[clave] = {
            "mAP50_95": round(float(m.map), 4),
            "mAP50": round(float(m.map50), 4),
            "precision": round(float(m.mp), 4),
            "recall": round(float(m.mr), 4),
        }
        print(f"    mAP50-95={m.map:.4f}  mAP50={m.map50:.4f}  "
              f"P={m.mp:.4f}  R={m.mr:.4f}\n")

    ind, cro = resultados["in_domain_camara"], resultados["cross_domain_movil"]
    brecha = {
        "delta_mAP50_95": round(ind["mAP50_95"] - cro["mAP50_95"], 4),
        "delta_mAP50": round(ind["mAP50"] - cro["mAP50"], 4),
        "delta_recall": round(ind["recall"] - cro["recall"], 4),
        "retencion_mAP50_95": (round(cro["mAP50_95"] / ind["mAP50_95"], 4)
                               if ind["mAP50_95"] else None),
    }
    reporte = {"modelo": str(model_path), "nombre": nombre, "imgsz": args.imgsz,
               "resultados": resultados, "brecha_de_dominio": brecha}

    destino = Path(args.json).resolve() if args.json else ds / f"crossdomain_{nombre}.json"
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 62)
    print(f"{'':22s} {'mAP50-95':>10s} {'mAP50':>8s} {'P':>8s} {'R':>8s}")
    for clave in ("in_domain_camara", "cross_domain_movil"):
        r = resultados[clave]
        print(f"{clave:22s} {r['mAP50_95']:10.4f} {r['mAP50']:8.4f} "
              f"{r['precision']:8.4f} {r['recall']:8.4f}")
    print(f"\nBrecha de dominio: mAP50-95 -{brecha['delta_mAP50_95']:.4f} "
          f"(retiene {brecha['retencion_mAP50_95']:.1%})"
          if brecha["retencion_mAP50_95"] is not None else "")
    print(f"Reporte: {destino}")


if __name__ == "__main__":
    main()
