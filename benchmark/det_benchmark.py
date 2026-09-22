#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark de DETECCION de placas con ultralytics.

Entrena cada modelo sobre det_data (train/val) y lo evalua sobre el split TEST
(held-out, anotado a mano en Roboflow). Guarda un JSON por modelo con:
  mAP50, mAP50-95, precision, recall, velocidad (ms/img).

Uso:
  python det_benchmark.py --data det_data/data.yaml --epochs 80 --out det_results
"""
import os, json, time, argparse, traceback

MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt",
    "rtdetr-l.pt",
]


def run_one(weights, data, epochs, imgsz, batch, workers, device, out, runs_dir, patience):
    from ultralytics import YOLO, RTDETR
    name = os.path.splitext(os.path.basename(weights))[0]
    Cls = RTDETR if name.startswith("rtdetr") else YOLO
    model = Cls(weights)
    t0 = time.time()
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
                device=device, project=runs_dir, name=name, exist_ok=True, verbose=False,
                plots=False, patience=patience)
    train_s = time.time() - t0
    # Evaluacion sobre el split TEST held-out
    m = model.val(data=data, split="test", imgsz=imgsz, batch=batch, workers=workers,
                  device=device, project=runs_dir, name=name + "_test", exist_ok=True,
                  plots=False, verbose=False)
    speed = getattr(m, "speed", {}) or {}
    res = {
        "name": name,
        "map50_95": round(float(m.box.map), 4),
        "map50": round(float(m.box.map50), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "ms_per_img": round(float(speed.get("inference", 0.0)), 3),
        "train_min": round(train_s / 60.0, 1),
        "epochs": epochs,
    }
    json.dump(res, open(os.path.join(out, name + ".json"), "w"), ensure_ascii=False, indent=2)
    print("%-12s  mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f  %.1fms  (train %.1f min)" %
          (name, res["map50"], res["map50_95"], res["precision"], res["recall"],
           res["ms_per_img"], res["train_min"]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="det_data/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="det_results")
    ap.add_argument("--runs", default="det_runs")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for w in a.models:
        try:
            run_one(w, a.data, a.epochs, a.imgsz, a.batch, a.workers, a.device, a.out, a.runs, a.patience)
        except Exception:
            print("ERROR en modelo %s:\n%s" % (w, traceback.format_exc()), flush=True)


if __name__ == "__main__":
    main()
