#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepara el deposito de Zenodo: estructura, rutas relativas, ZIPs y sumas.

Produce en --out una carpeta lista para arrastrar a Zenodo:

    README.md                 visible en la ficha, sin descargar nada
    USE_AGREEMENT.md          condiciones de acceso
    dataset_metadata.json     conteos, rango temporal y sumas SHA-256
    recognition_crops.zip     579 292 recortes de matricula
    evaluation_sets.zip       Gold-A, Gold-B, retenido, ajuste fino y sus TSV
    detection_2600.zip        2 600 imagenes con la caja anotada a mano

Los TSV se reescriben con rutas relativas al deposito: las originales apuntan a
F:\\DATAPLACAS y no sirven fuera de esta maquina. El script verifica que cada
fichero referenciado existe antes de empaquetar y aborta si falta alguno.

Uso (desde benchmark/):
    python prepare_zenodo.py --dry-run      # solo verifica y cuenta
    python prepare_zenodo.py                # construye el deposito
"""
import os
import csv
import json
import shutil
import hashlib
import zipfile
import argparse
import datetime

BENCH = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BENCH)

CROPS = os.path.join(RAIZ, "Peruvian_ALPR_Dataset_final", "crops")
GOLDA = os.path.join(RAIZ, "dataset_nuevo_validacion", "images_plates")
DET = os.path.join(BENCH, "det_data")

# tsv de origen -> (nombre en el deposito, descripcion)
TSVS = [
    ("data/rec_labels_train.tsv", "train.tsv", "particion de entrenamiento"),
    ("data/rec_labels_val.tsv", "val.tsv", "particion de validacion"),
    ("data/rec_labels_test.tsv", "test.tsv", "particion de prueba"),
    ("data/rec_labels_gold.tsv", "gold_A.tsv", "Gold-A, vista por evento"),
    ("data/rec_labels_gold2.tsv", "gold_B.tsv", "Gold-B, vista por matricula"),
    ("data/rec_labels_gold2_clean.tsv", "gold_B_heldout.tsv", "retenido, control de fuga"),
    ("data/rec_labels_gold_all.tsv", "finetune.tsv", "conjunto de ajuste fino"),
]


def rel(ruta):
    """Traduce una ruta absoluta de Windows a la del deposito."""
    p = ruta.replace("\\", "/")
    for marca, destino in (("/Peruvian_ALPR_Dataset_final/crops/", "crops/"),
                           ("/dataset_nuevo_validacion/images_plates/", "gold_A_crops/")):
        i = p.find(marca)
        if i >= 0:
            return destino + p[i + len(marca):]
    return None


def origen(ruta):
    """Ruta real en disco de una entrada del TSV."""
    p = ruta.replace("\\", "/")
    for marca, base in (("/Peruvian_ALPR_Dataset_final/crops/", CROPS),
                        ("/dataset_nuevo_validacion/images_plates/", GOLDA)):
        i = p.find(marca)
        if i >= 0:
            return os.path.join(base, p[i + len(marca):].replace("/", os.sep))
    return None


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def zipear(destino, entradas, etiqueta):
    """entradas: lista de (ruta_en_disco, nombre_dentro_del_zip)."""
    n = len(entradas)
    print("  %s: %d ficheros ..." % (etiqueta, n), flush=True)
    # ZIP_STORED: los JPEG ya estan comprimidos, deflate no gana y tarda mucho.
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, (src, arc) in enumerate(entradas, 1):
            z.write(src, arc)
            if k % 50000 == 0:
                print("     %d/%d" % (k, n), flush=True)
    mb = os.path.getsize(destino) / 1024 ** 2
    print("  -> %s  %.0f MB" % (os.path.basename(destino), mb), flush=True)
    return {"fichero": os.path.basename(destino), "ficheros": n,
            "MB": round(mb, 1), "sha256": sha256(destino)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "zenodo_peruvian_alpr"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("verificando fuentes ...", flush=True)
    for d in (CROPS, GOLDA, DET):
        if not os.path.isdir(d):
            raise SystemExit("falta la carpeta %s" % d)

    # Indice de nombres por carpeta. Un listdir por directorio en vez de medio
    # millon de os.path.exists, que sobre disco de red tarda una eternidad.
    indice = {}
    for sub in ("train", "val", "test"):
        d = os.path.join(CROPS, sub)
        indice["crops/%s/" % sub] = set(os.listdir(d)) if os.path.isdir(d) else set()
    indice["gold_A_crops/"] = set(os.listdir(GOLDA))
    print("  índice: " + ", ".join("%s %d" % (k, len(v)) for k, v in indice.items()), flush=True)

    def existe(relativa):
        i = relativa.rfind("/") + 1
        return relativa[i:] in indice.get(relativa[:i], ())

    # ---------------------------------------------- TSV y ficheros referenciados
    tsv_out, necesarios, faltan = {}, set(), []
    for src, nombre, _ in TSVS:
        p = os.path.join(BENCH, src)
        if not os.path.exists(p):
            raise SystemExit("falta %s" % p)
        filas = []
        with open(p, encoding="utf-8") as f:
            for linea in f:
                linea = linea.rstrip("\n")
                if not linea:
                    continue
                ruta, placa = linea.split("\t")
                r, o = rel(ruta), origen(ruta)
                if r is None or o is None:
                    faltan.append(("ruta no reconocida", ruta))
                    continue
                if not existe(r):
                    faltan.append(("no existe", o))
                    continue
                filas.append((r, placa))
                necesarios.add((o, r))
        tsv_out[nombre] = filas
        print("  %-20s %6d entradas" % (nombre, len(filas)), flush=True)

    if faltan:
        print("\n%d problemas, los primeros:" % len(faltan))
        for m, r in faltan[:10]:
            print("   %s: %s" % (m, r))
        raise SystemExit("abortado: hay entradas sin fichero en disco")

    crops = sorted(x for x in necesarios if x[1].startswith("crops/"))
    golda = sorted(x for x in necesarios if x[1].startswith("gold_A_crops/"))
    det = []
    for sub in ("train", "val", "test"):
        for k in ("images", "labels"):
            d = os.path.join(DET, k, sub)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    det.append((os.path.join(d, f), "%s/%s/%s" % (k, sub, f)))
    det.append((os.path.join(DET, "data.yaml"), "data.yaml"))

    print("\nrecortes de reconocimiento : %d" % len(crops))
    print("recortes de Gold-A         : %d" % len(golda))
    print("ficheros de detección      : %d" % len(det))

    if a.dry_run:
        print("\ndry-run: no se escribió nada")
        return

    # -------------------------------------------------------------- construir
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp_labels")
    os.makedirs(tmp, exist_ok=True)
    for nombre, filas in tsv_out.items():
        with open(os.path.join(tmp, nombre), "w", encoding="utf-8", newline="") as f:
            for r, placa in filas:
                f.write("%s\t%s\n" % (r, placa))

    print("\nempaquetando ...", flush=True)
    resumen = {}
    resumen["recognition_crops"] = zipear(
        os.path.join(a.out, "recognition_crops.zip"), crops, "recortes de reconocimiento")

    eval_entradas = list(golda) + [(os.path.join(tmp, n), "labels/" + n) for n in tsv_out]
    resumen["evaluation_sets"] = zipear(
        os.path.join(a.out, "evaluation_sets.zip"), eval_entradas, "conjuntos de evaluación")

    resumen["detection_2600"] = zipear(
        os.path.join(a.out, "detection_2600.zip"), det, "detección")

    shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------- metadatos
    meta = {
        "titulo": "Peruvian ALPR Plate Crops (PAPC-1)",
        "version": "1.0",
        "fecha_deposito": datetime.date.today().isoformat(),
        "periodo_captura": "2025-03-07 / 2026-06-15",
        "recortes_reconocimiento": len(crops),
        "matriculas_unicas": 29928,
        "imagenes_deteccion": sum(1 for _, n in det if n.startswith("images/")),
        "particiones": {n: len(v) for n, v in tsv_out.items()},
        "acceso": "restringido, bajo acuerdo de uso",
        "licencia": "CC BY-NC 4.0 con acuerdo de uso adicional",
        "paquetes": resumen,
    }
    with open(os.path.join(a.out, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    escribir_readme(a.out, meta)
    escribir_acuerdo(a.out)

    print("\ndepósito listo en %s" % a.out)
    print("Sube los 6 ficheros a Zenodo. README.md y USE_AGREEMENT.md van sueltos,")
    print("para que se lean en la ficha sin descargar nada.")


def escribir_readme(out, m):
    p = m["particiones"]
    txt = f"""# Peruvian ALPR Plate Crops (PAPC-1)

Recortes de matrícula peruana capturados por una cámara IP fija en el ingreso
vehicular de una universidad del altiplano (Juliaca, Puno, >3800 m s.n.m.),
en operación continua entre {m['periodo_captura'].replace(' / ', ' y ')}.

Hasta donde alcanza nuestra revisión no existe otro conjunto público de
matrículas peruanas ni de la región andina de magnitud comparable, ni capturado
en operación continua prolongada en lugar de campañas puntuales.

## Contenido

| fichero | contenido |
|---|---|
| `recognition_crops.zip` | {m['recortes_reconocimiento']:,} recortes de matrícula, de {m['matriculas_unicas']:,} placas únicas |
| `evaluation_sets.zip` | conjuntos de evaluación verificados y todas las etiquetas |
| `detection_2600.zip` | {m['imagenes_deteccion']:,} imágenes con la caja de la matrícula anotada a mano |

### Etiquetas

`evaluation_sets.zip` contiene `labels/*.tsv`, con dos columnas separadas por
tabulador: ruta relativa al depósito y texto de la matrícula sin guion.

| fichero | n | papel |
|---|---|---|
| `train.tsv` | {p.get('train.tsv', 0):,} | entrenamiento, etiqueta *silver* |
| `val.tsv` | {p.get('val.tsv', 0):,} | validación, etiqueta *silver* |
| `test.tsv` | {p.get('test.tsv', 0):,} | prueba, etiqueta *silver* |
| `gold_A.tsv` | {p.get('gold_A.tsv', 0):,} | verdad-terreno, muestreo por acceso registrado |
| `gold_B.tsv` | {p.get('gold_B.tsv', 0):,} | verdad-terreno, una entrada por matrícula. **Evaluación principal** |
| `gold_B_heldout.tsv` | {p.get('gold_B_heldout.tsv', 0):,} | subconjunto de Gold-B sin solape con el ajuste fino |
| `finetune.tsv` | {p.get('finetune.tsv', 0):,} | conjunto de ajuste fino, elegido por selección activa |

Las particiones están **agrupadas por matrícula**: ninguna placa aparece en dos
de ellas. Un corte aleatorio por fotograma habría situado imágenes casi idénticas
del mismo vehículo a ambos lados de la partición.

Las etiquetas de `train`, `val` y `test` son *silver*: proceden de un OCR de
propósito general corregido con reglas de posición, no de anotación humana. Las
de los conjuntos `gold_*` se verificaron contra el registro vehicular oficial.

### Nombres de fichero

Los recortes codifican metadatos en el nombre:

    <marca_temporal>_<confianza>_(<alto>, <ancho>, 3)_<placa>.jpg

La marca temporal permite reconstruir la condición lumínica por elevación solar
real y la frecuencia de paso de cada vehículo.

### Detección

`detection_2600.zip` sigue el formato YOLO (una clase, caja alineada a los ejes)
con `data.yaml` incluido. Aviso: su partición procede de un reparto aleatorio por
archivo, de modo que hay solape temporal entre subconjuntos. Sirve para entrenar
un detector, no para estimar su generalización.

## Cómo se construyó

Cámara Hikvision DS-2CD2683G2-LIZS2U a 2.50 m sobre el carril, operando a
1280×720 px y 10 fps. Detección con YOLOv8n, lectura con PaddleOCR PP-OCRv4 más
un corrector determinista por posición para el patrón nacional `LNN-DDD`, y
filtros de deduplicación, validación de formato y área mínima.

El recorte mediano mide 51×101 px. Es un régimen de objeto pequeño: cada carácter
dispone de unos 17 píxeles de ancho, insuficientes para segmentar de forma fiable.

## Privacidad

El encuadre excluye por diseño a los ocupantes del vehículo. Se realizó además un
barrido automático de rostros sobre la totalidad del corpus y los 6357 fotogramas
(1.10 %) con peatones identificables se difuminaron antes de recortar.

**Solo se depositan los recortes de la región de la matrícula.** Los fotogramas
completos no se publican en ninguna modalidad, porque pueden contener peatones y
vehículos ajenos al estudio.

## Acceso y uso

Acceso restringido: la matrícula es un dato personal vinculable a su titular a
través del registro vehicular oficial, y su tratamiento se rige por la Ley 29733
de protección de datos personales del Perú. Lee `USE_AGREEMENT.md` antes de
solicitar acceso.

## Cita

    Condori-Coaquira, A.R. (2026). Peruvian ALPR Plate Crops (PAPC-1)
    [conjunto de datos]. Zenodo. https://doi.org/[DOI]

## Contacto

Angel Rosendo Condori-Coaquira, Facultad de Ingeniería, Universidad Peruana
Unión, Juliaca, Puno, Perú. angelrc2@upeu.edu.pe
ORCID 0000-0003-4681-4272
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(txt)


def escribir_acuerdo(out):
    txt = """# Acuerdo de uso

Este conjunto contiene matrículas de vehículos reales. En el Perú la matrícula es
un dato personal, porque permite identificar a su titular a través del registro
vehicular oficial, y su tratamiento se rige por la Ley 29733 de protección de
datos personales. Por eso el acceso es restringido.

Al solicitar acceso, quien lo hace se compromete a:

1. **Usar los datos solo con fines académicos o de investigación no comercial.**
   Queda excluido cualquier uso comercial, operativo o de vigilancia.

2. **No intentar reidentificar a los titulares** de los vehículos, ni cruzar
   estos datos con registros vehiculares, padrones u otras fuentes que permitan
   asociar una matrícula a una persona.

3. **No redistribuir los datos**, ni completos ni en parte, por ningún medio.
   Quien los necesite debe solicitarlos por esta misma vía.

4. **No publicar imágenes individuales** con la matrícula legible en artículos,
   presentaciones o repositorios. En las figuras, cubrir al menos un carácter.

5. **Citar el conjunto y el artículo asociado** en cualquier trabajo derivado.

6. **Comunicar al autor de correspondencia** las publicaciones que lo utilicen,
   para poder mantener el registro de usos.

7. **Eliminar los datos** al concluir la investigación para la que se solicitaron.

## Cómo solicitar acceso

Desde la propia ficha de Zenodo, con el botón de solicitud. En la justificación
indica tu afiliación institucional, el propósito de la investigación y el uso
previsto. Las solicitudes sin afiliación verificable o sin propósito concreto no
se atienden.

## Qué no está incluido

Los fotogramas completos de los que proceden los recortes no forman parte del
depósito ni se facilitan bajo solicitud, porque pueden contener peatones y
vehículos ajenos al estudio.
"""
    with open(os.path.join(out, "USE_AGREEMENT.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
