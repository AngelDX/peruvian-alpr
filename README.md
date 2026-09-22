# Reconocimiento de matrículas peruanas en el borde

Código de análisis del artículo *Reconocimiento de matrículas peruanas en el borde:
modelos LPRNet y CRNN destilados de un OCR genérico con corrección determinista por
posición*, enviado a **Sensors** (MDPI).

Un sistema ALPR con cámara IP fija operó sin interrupción durante 16 meses en el
ingreso vehicular de una universidad del altiplano peruano y registró 579 292 recortes
de matrícula de 29 928 placas únicas. Sobre ese corpus se destilan reconocedores
compactos a partir de un OCR genérico corregido con reglas de posición, y se estudia
bajo qué condiciones de evaluación la conclusión cambia de signo.

## Qué contiene y qué no

Este repositorio contiene **solo código y resultados agregados**. No incluye imágenes,
etiquetas ni ninguna matrícula.

La matrícula es un dato personal vinculable a su titular a través del registro
vehicular oficial, y su tratamiento se rige por la Ley 29733 del Perú. Por eso el
conjunto se deposita aparte, en Zenodo, con acceso mediante acuerdo de uso:

> DOI del conjunto: [10.5281/zenodo.22021714](https://doi.org/10.5281/zenodo.22021714)

Para reproducir cualquier cifra hay que solicitar acceso al depósito, descargarlo y
apuntar los scripts a la carpeta resultante.

## Estructura

```
benchmark/                     scripts de análisis
├── bench_utils.py             charset, codificación CTC, métricas
├── corrector.py               corrector determinista por posición (LNN-DDD)
├── datasets.py                Dataset PyTorch de recortes
├── models.py                  LPRNet y CRNN con pérdida CTC
├── train.py                   destilación con etiquetas silver
├── finetune.py                ajuste fino con verdad-terreno, tres papeles disjuntos
├── cv_finetune.py             validación cruzada agrupada por matrícula
├── analisis_inferencial.py    Q de Cochran, McNemar, Holm, matriz por placa
├── regenerar_cifras_32x100.py regenera todas las cifras de la Sección 4.3
├── fig_*.py                   figuras del manuscrito
└── ablation_resolucion/       ablación de la resolución de normalización
results/
├── cifras_32x100.json         cifras agregadas del artículo
└── latex/                     tablas generadas, listas para el manuscrito
docs/
└── REPRODUCIR.md              qué script produce cada tabla y figura
```

## Requisitos

```bash
pip install -r requirements.txt
```

Probado con Python 3.12, PyTorch 2.13 con CUDA y Ultralytics 8.4 sobre una NVIDIA
RTX 5060 Ti. La inferencia funciona en CPU; el entrenamiento completo, no de forma
razonable.

## Reproducir

Con el conjunto descargado de Zenodo y descomprimido:

```bash
cd benchmark

# 1. Destilación con etiquetas silver, tres semillas
python ablation_resolucion/run_ablation.py --only 32x100 --epochs 10 --seed 0
python ablation_resolucion/run_ablation.py --only 32x100 --epochs 10 --seed 1
python ablation_resolucion/run_ablation.py --only 32x100 --epochs 10 --seed 2

# 2. Ajuste fino de ambas arquitecturas sobre las tres semillas
python ablation_resolucion/finetune_all.py

# 3. Todas las cifras de la Sección 4.3
python regenerar_cifras_32x100.py

# 4. Figuras
python fig_arquitectura_sistema.py     # Figura 1
python fig_perfiles.py                 # Figura 5
python fig_posthoc_cajas.py            # Figura 6
```

`docs/REPRODUCIR.md` detalla qué script produce cada tabla y figura del artículo.

## Notas de reproducibilidad

**La variabilidad entre semillas no es despreciable.** Es de 0.57 a 0.84 puntos en
LPRNet y de 1.48 a 1.51 en el CRNN. Cualquier diferencia menor de un punto entre dos
corridas individuales debe atribuirse a ruido, no al modelo.

**Los modelos sobreajustan pronto.** La exactitud de validación alcanza su máximo entre
las épocas dos y siete y después desciende, mientras la pérdida de entrenamiento sigue
cayendo. Los scripts conservan el mejor punto de control, nunca el último.

**Cuidado al estratificar por área en Gold-B.** Ese conjunto sobremuestrea la noche a
propósito, y el área está correlacionada con la condición lumínica: su cuartil de
matrículas grandes contiene un 34 % de capturas nocturnas frente al 11 % del cuartil
pequeño. El análisis de tamaño va sobre Gold-A.

## Cita

Si usas este código, cita el artículo y el conjunto de datos:

```bibtex
@article{CondoriCoaquira2026alpr,
  author  = {Condori-Coaquira, Angel Rosendo and Tocto-Cano, Esteban and
             Humpiri-Flores, Milton Edward and Mamani-Pari, David},
  title   = {Peruvian License Plate Recognition at the Edge: Compact LPRNet and
             CRNN Models Distilled from a Rule-Corrected Generic OCR},
  journal = {Sensors},
  year    = {2026},
  note    = {En revisión}
}

@dataset{CondoriCoaquira2026papc,
  author    = {Condori-Coaquira, Angel Rosendo},
  title     = {Peruvian ALPR Plate Crops (PAPC-1)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22021714}
}
```

## Licencia

Código bajo licencia MIT, en `LICENSE`. El conjunto de datos se rige por su propio
acuerdo de uso, en el depósito de Zenodo.

## Contacto

Angel Rosendo Condori-Coaquira, Facultad de Ingeniería, Universidad Peruana Unión,
Juliaca, Puno, Perú. angelrc2@upeu.edu.pe ·
[ORCID 0000-0003-4681-4272](https://orcid.org/0000-0003-4681-4272)
