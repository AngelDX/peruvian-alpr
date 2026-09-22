#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrena el detector movil de placas de auto sobre MobileALPR_dataset y lo evalua
en los dos dominios por separado.

La augmentacion geometrica se sube por encima de los valores por defecto de Ultralytics
(rotacion, perspectiva, escala) de forma deliberada: el dominio de la camara fija no tiene
variacion natural de angulo ni de distancia focal, asi que hay que inducirla para que el
detector no vuelva a memorizar el encuadre de la instalacion.

Uso:
    python train_mobile_detector.py --model yolo11n.pt --epochs 80
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt", help="pesos base (n = tamano movil)")
    ap.add_argument("--dataset", default="../MobileALPR_dataset")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="nombre de la corrida")
    ap.add_argument("--project", default="det_runs_mobile")
    args = ap.parse_args()

    from ultralytics import YOLO

    ds = Path(args.dataset).resolve()
    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"Falta {data_yaml}; genera el dataset con build_mobile_dataset.py")
    nombre = args.name or Path(args.model).stem + "_mobile"

    modelo = YOLO(args.model)
    print(f"Entrenando {args.model} -> {args.project}/{nombre}")
    modelo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=nombre,
        exist_ok=True,
        verbose=True,
        plots=True,
        # Augmentacion reforzada para romper el prior de la camara fija.
        degrees=10.0,        # la camara siempre esta nivelada; un celular no
        perspective=0.0005,  # angulos oblicuos, ausentes en la instalacion
        scale=0.6,           # distancia al vehiculo mucho mas variable
        translate=0.15,      # la placa ya no cae siempre en la misma region
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )

    # Ultralytics resuelve `project` contra su propio runs_dir configurado, que puede
    # estar en otra unidad; hay que preguntarle al trainer en vez de reconstruir la ruta.
    best = Path(modelo.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nMejor checkpoint: {best}")

    # --- Evaluacion en ambos dominios, por separado
    entrenado = YOLO(str(best))
    resultados = {}
    for clave, yml in (("in_domain_camara", "data_test_camara.yaml"),
                       ("cross_domain_movil", "data_test_movil.yaml")):
        m = entrenado.val(data=str(ds / yml), split="test", imgsz=args.imgsz,
                          verbose=False, plots=False).box
        resultados[clave] = {
            "mAP50_95": round(float(m.map), 4),
            "mAP50": round(float(m.map50), 4),
            "precision": round(float(m.mp), 4),
            "recall": round(float(m.mr), 4),
        }

    reporte = {
        "modelo_base": args.model,
        "checkpoint": str(best.resolve()),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed,
        "resultados": resultados,
    }
    destino = ds / f"crossdomain_{nombre}.json"
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'':22s} {'mAP50-95':>10s} {'mAP50':>8s} {'P':>8s} {'R':>8s}")
    for k, r in resultados.items():
        print(f"{k:22s} {r['mAP50_95']:10.4f} {r['mAP50']:8.4f} "
              f"{r['precision']:8.4f} {r['recall']:8.4f}")
    print(f"Reporte: {destino}")


if __name__ == "__main__":
    main()
