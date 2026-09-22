#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalcula la clasificacion luminica de cada frame con la posicion real del Sol.

PROBLEMA. La columna `day_night` del corpus es una regla de reloj fija (06:00-17:59 = dia).
En Juliaca (15.5 S) el ocaso se desplaza unos 70 minutos a lo largo del año -- de las 17:25
en junio a las 18:35 en diciembre -- de modo que la hora 17:00 en invierno austral queda
ya oscura pero etiquetada como "dia". Eso subestima la fraccion nocturna del conjunto.

SOLUCION. Se calcula la elevacion solar real para el instante y el lugar de cada frame
(algoritmo NOAA) y se clasifica en tres clases con significado fotometrico:

    dia         elevacion >  0 grados   Sol sobre el horizonte
    crepusculo  -6 < elev <= 0          crepusculo civil: hay luz, pero decae rapido
    noche       elevacion <= -6         oscuridad efectiva, depende de luz artificial

Las tres clases son mas informativas que el binario original: el crepusculo es un regimen
propio, ni diurno ni nocturno, y en este emplazamiento concentra buena parte del transito.

No se modifica ningun archivo existente: se escribe una metadata enriquecida aparte.

Uso:
    python recompute_daynight.py --lat -15.50 --lon -70.13
"""
import argparse, csv, gzip, json, math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def elevacion_solar(dt_utc, lat_deg, lon_deg):
    """Elevacion del Sol en grados (algoritmo NOAA, sin correccion de refraccion)."""
    # Dia juliano
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jdn = (dt_utc.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045)
    frac = (dt_utc.hour - 12) / 24 + dt_utc.minute / 1440 + dt_utc.second / 86400
    jd = jdn + frac
    t = (jd - 2451545.0) / 36525.0   # siglos julianos desde J2000

    # Geometria orbital
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_rad = math.radians(m_anom)
    centro = (math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
              + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
              + math.sin(3 * m_rad) * 0.000289)
    lon_verdadera = l0 + centro
    omega = 125.04 - 1934.136 * t
    lon_aparente = lon_verdadera - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Oblicuidad y declinacion
    eps0 = (23 + (26 + ((21.448 - t * (46.815 + t * (0.00059 - t * 0.001813)))) / 60) / 60)
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps))
                                  * math.sin(math.radians(lon_aparente))))

    # Ecuacion del tiempo (minutos)
    var_y = math.tan(math.radians(eps / 2)) ** 2
    l0_rad = math.radians(l0)
    eq_tiempo = 4 * math.degrees(
        var_y * math.sin(2 * l0_rad)
        - 2 * e * math.sin(m_rad)
        + 4 * e * var_y * math.sin(m_rad) * math.cos(2 * l0_rad)
        - 0.5 * var_y * var_y * math.sin(4 * l0_rad)
        - 1.25 * e * e * math.sin(2 * m_rad)
    )

    # Angulo horario
    minutos = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60
    tiempo_solar = (minutos + eq_tiempo + 4 * lon_deg) % 1440
    angulo_h = tiempo_solar / 4 - 180 if tiempo_solar / 4 >= 0 else tiempo_solar / 4 + 180

    lat_rad = math.radians(lat_deg)
    decl_rad = math.radians(decl)
    cos_z = (math.sin(lat_rad) * math.sin(decl_rad)
             + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(math.radians(angulo_h)))
    cos_z = max(-1.0, min(1.0, cos_z))
    return 90.0 - math.degrees(math.acos(cos_z))


def clasificar(elev):
    if elev > 0:
        return "dia"
    if elev > -6:
        return "crepusculo"
    return "noche"


def autocomprobacion(lat, lon, tz):
    """Ocaso aproximado en varias fechas: debe rondar 17:25 en junio y 18:35 en diciembre."""
    print("Autocomprobacion — hora local del ocaso (elevacion cruza 0):")
    for fecha in ("2025-06-21", "2025-09-21", "2025-12-21", "2026-03-21"):
        d = datetime.strptime(fecha, "%Y-%m-%d")
        anterior, hora_ocaso = None, None
        for minuto in range(15 * 60, 20 * 60):   # busca entre las 15:00 y las 20:00 locales
            local = d + timedelta(minutes=minuto)
            utc = local.replace(tzinfo=timezone(timedelta(hours=tz))).astimezone(timezone.utc)
            elev = elevacion_solar(utc, lat, lon)
            if anterior is not None and anterior > 0 >= elev:
                hora_ocaso = local
                break
            anterior = elev
        print(f"  {fecha}: {hora_ocaso.strftime('%H:%M') if hora_ocaso else '—'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="../Peruvian_ALPR_Dataset_final/metadata_full.csv.gz")
    ap.add_argument("--out", default="../Peruvian_ALPR_Dataset_final/metadata_full_light.csv.gz")
    ap.add_argument("--reporte", default="../docs/paper_materials/11_clasificacion_luz.json")
    ap.add_argument("--lat", type=float, default=-15.50, help="latitud del emplazamiento")
    ap.add_argument("--lon", type=float, default=-70.13, help="longitud del emplazamiento")
    ap.add_argument("--tz", type=float, default=-5.0, help="huso horario local (Peru = -5)")
    args = ap.parse_args()

    autocomprobacion(args.lat, args.lon, args.tz)
    print()

    src = Path(args.metadata).resolve()
    dst = Path(args.out).resolve()
    matriz = Counter()      # (etiqueta original, clase nueva)
    por_hora = Counter()    # (hora local, clase nueva)
    clases = Counter()
    n = 0

    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f_in, \
         gzip.open(dst, "wt", encoding="utf-8", newline="") as f_out:
        lector = csv.DictReader(f_in)
        campos = lector.fieldnames + ["sun_elev_deg", "light_class"]
        escritor = csv.DictWriter(f_out, fieldnames=campos)
        escritor.writeheader()
        for row in lector:
            dt_utc = datetime.fromtimestamp(float(row["timestamp"]), tz=timezone.utc)
            elev = elevacion_solar(dt_utc, args.lat, args.lon)
            clase = clasificar(elev)
            row["sun_elev_deg"] = f"{elev:.2f}"
            row["light_class"] = clase
            escritor.writerow(row)
            matriz[(row["day_night"], clase)] += 1
            por_hora[(int(row["hour"]), clase)] += 1
            clases[clase] += 1
            n += 1

    print(f"Frames procesados: {n}")
    print(f"\nClasificacion nueva:")
    for c in ("dia", "crepusculo", "noche"):
        print(f"  {c:11s} {clases[c]:7d}  {100*clases[c]/n:5.2f} %")

    print(f"\nMatriz etiqueta original -> clase nueva:")
    print(f"  {'original':10s} {'dia':>9s} {'crepusculo':>11s} {'noche':>9s}")
    for orig in ("dia", "noche"):
        fila = "  ".join(f"{matriz[(orig, c)]:>9d}" for c in ("dia", "crepusculo", "noche"))
        print(f"  {orig:10s} {fila}")

    reclasificados = matriz[("dia", "crepusculo")] + matriz[("dia", "noche")]
    print(f"\nFrames que la regla de reloj daba por 'dia' y no lo eran: {reclasificados}"
          f"  ({100*reclasificados/n:.2f} %)")
    print(f"Frames 'noche' por reloj que tenian el Sol arriba: {matriz[('noche', 'dia')]}")

    print(f"\nHoras con mas de una clase (la regla de reloj no puede separarlas):")
    horas = sorted({h for h, _ in por_hora})
    for h in horas:
        cs = {c: por_hora[(h, c)] for c in ("dia", "crepusculo", "noche") if por_hora[(h, c)]}
        if len(cs) > 1:
            print(f"  {h:02d}:00  " + "  ".join(f"{c}={v}" for c, v in cs.items()))

    reporte = {
        "emplazamiento": {"lat": args.lat, "lon": args.lon, "tz": args.tz},
        "definicion": {
            "dia": "elevacion solar > 0 grados",
            "crepusculo": "-6 < elevacion <= 0 (crepusculo civil)",
            "noche": "elevacion <= -6 grados",
        },
        "n_frames": n,
        "clases": dict(clases),
        "proporciones": {c: clases[c] / n for c in clases},
        "matriz_original_vs_nueva": {f"{a}->{b}": v for (a, b), v in sorted(matriz.items())},
        "regla_original": "day_night por corte horario: 06:00-17:59 = dia",
        "metadata_enriquecida": str(dst),
    }
    rep = Path(args.reporte).resolve()
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetadata enriquecida: {dst}")
    print(f"Reporte: {rep}")


if __name__ == "__main__":
    main()
