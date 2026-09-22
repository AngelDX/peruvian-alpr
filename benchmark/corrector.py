# -*- coding: utf-8 -*-
"""Corrector de placas por posicion (espejo de app_server_dataset/ocr.py).

Intenta importar la funcion real de produccion; si no esta disponible, usa una
copia local identica de las reglas (formato peruano + O/0, I/1 medidas en SUNARP).
"""
import os, sys

def _load_production():
    for cand in [os.path.join(os.path.dirname(__file__), "..", "app_server_dataset"),
                 os.path.join(os.path.dirname(__file__), "app_server_dataset")]:
        cand = os.path.abspath(cand)
        if os.path.isfile(os.path.join(cand, "ocr.py")):
            sys.path.insert(0, cand)
            try:
                from ocr import correct_plate6 as f
                return f
            except Exception:
                pass
    return None

_D2L = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "Z", "8": "B"}
_L2D = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
        "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}

def _local_correct6(p6):
    if len(p6) != 6:
        return p6
    c = list(p6)
    if c[0].isdigit():
        c[0] = _D2L.get(c[0], c[0])
    for i in (3, 4, 5):
        if c[i].isalpha():
            c[i] = _L2D.get(c[i], c[i])
    if c[1] in ("O", "0"): c[1] = "0"
    if c[1] in ("I", "1"): c[1] = "1"
    if c[2] in ("O", "0"): c[2] = "O"
    if c[2] in ("I", "1"): c[2] = "I"
    return "".join(c)

_prod = _load_production()

def correct(plate):
    """Aplica el corrector a una placa (con o sin guion). Devuelve 6 chars sin guion."""
    p6 = "".join(ch for ch in plate.upper() if ch.isalnum())
    if len(p6) != 6:
        return p6
    return (_prod(p6) if _prod else _local_correct6(p6))
