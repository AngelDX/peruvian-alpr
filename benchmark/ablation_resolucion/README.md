# Ablación de la resolución de normalización de la ROI

## Por qué

La matrícula peruana es casi exactamente **2:1**. Medido sobre los 439 178 recortes
de `data/rec_labels_train.tsv` cuyo tamaño nativo va codificado en el nombre del
archivo:

| | p1 | p5 | p25 | mediana | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| alto | 34 | 39 | 45 | **51** | 59 | 83 | 133 |
| ancho | 82 | 85 | 92 | **101** | 110 | 124 | 139 |
| aspecto | 0.95 | 1.43 | 1.85 | **1.98** | 2.08 | 2.26 | 2.56 |

Las dos resoluciones que usa hoy el benchmark distorsionan ese aspecto:

- LPRNet a **24 × 94** da 3.92:1 y comprime la altura de 51 px a 24.
- CRNN a **32 × 100** da 3.12:1.
- El 96 × 320 del artículo coreano da 3.33:1 y viene de placas con otra proporción.

La hipótesis es que ese aplastamiento vertical contribuye al fallo documentado en
la sección 4.12 del manuscrito, donde LPRNet devuelve lecturas de cinco caracteres
en 24 de 1130 casos y 11 de ellas son omisiones en las posiciones 2 o 3, un patrón
compatible con fusión de activaciones adyacentes en la decodificación CTC codiciosa.

## Qué se compara

Seis resoluciones por modelo. `48 × 96` es la candidata: respeta el 2:1 y es
prácticamente el tamaño nativo mediano, así que solo amplía el 39 % de los recortes
frente al 97 % de 64 × 128 y el 100 % de 96 × 320.

| entrada | aspecto | recortes ampliados | papel |
|---|---|---|---|
| 24 × 94 | 3.92 | 28 % | línea base LPRNet |
| 32 × 100 | 3.12 | 45 % | línea base CRNN |
| **48 × 96** | **2.00** | **39 %** | **nativa 2:1** |
| 64 × 128 | 2.00 | 97 % | 2:1 ampliada |
| 80 × 160 | 2.00 | 100 % | 2:1 muy ampliada |
| 96 × 320 | 3.33 | 100 % | referencia coreana |

## Archivos

- `models_res.py` — LPRNet y CRNN que aceptan cualquier `(H, W)`. Único cambio
  respecto de `../models.py`: la altura se colapsa con un pooling adaptativo en vez
  de uno de tamaño fijo. **A 24 × 94 y 32 × 100 el grafo es idéntico al original**,
  así que la fila de línea base debe reproducir las cifras ya publicadas. Si no lo
  hace, algo más cambió y conviene averiguar qué antes de seguir.
- `run_ablation.py` — entrena, elige época por validación y evalúa en Gold-B.
  Guarda las predicciones por matrícula para poder hacer contrastes pareados.
- `report_ablation.py` — tabla LaTeX, figura y McNemar exacta con corrección de Holm.

## Cómo correrlo

Todo desde `benchmark/`, que es donde están `datasets.py` y `bench_utils.py`.

```bash
# 1. Comprobar que los modelos construyen bien a las seis resoluciones (segundos)
python ablation_resolucion/models_res.py

# 2. Humo: 1 época, 2000 recortes, 12 configuraciones. Unos 10 minutos.
python ablation_resolucion/run_ablation.py --smoke

# 3. Corrida real
python ablation_resolucion/run_ablation.py --epochs 30 --seed 0

# 4. Informe
python ablation_resolucion/report_ablation.py
```

### Qué imprime

Por época: pérdida, exactitud de validación, CER, **tasa de lecturas de longitud
distinta de seis**, tasa de aprendizaje, segundos y ETA, marcando la mejor época.
Cada cinco épocas vuelca seis fallos de ejemplo (`--muestras 0` lo desactiva).
Al cerrar cada configuración imprime la tabla acumulada ordenada por Gold-B, así
que puedes cortar en cuanto veas la tendencia.

Todo va también a `resultados/logs/<modelo>_<HxW>_s<semilla>.log` sin buffering,
y el historial por época queda dentro del JSON.

El JSON se reescribe después de cada configuración, así que si se corta la corrida
no pierdes lo ya hecho y puedes retomar con `--only`:

```bash
python ablation_resolucion/run_ablation.py --epochs 30 --only 80x160,96x320
```

### Cuánto tarda

12 configuraciones sobre los 100 000 recortes de `data_sub`. En una RTX 5060 Ti
calculo entre 4 y 7 horas para las 30 épocas, con 96 × 320 como la más lenta por
tener T = 80. Si quieres acortarlo, `--models lprnet` primero: es el modelo
desplegable y el que muestra el fallo de longitud.

### Semillas

Tu propia validación cruzada da una desviación entre pliegues de 0.45 a 0.53
puntos, así que **cualquier diferencia menor de un punto entre dos resoluciones es
ruido**. Si el resultado sale ajustado, repite con `--seed 1` y `--seed 2`;
`report_ablation.py` detecta los tres JSON, promedia y dibuja las barras de error.

## Qué mirar en los resultados

1. ¿Gana alguna resolución 2:1 sobre la línea base, y por más de un punto?
2. ¿Baja la tasa de lecturas de longitud distinta de seis? Es la predicción
   concreta de la hipótesis y el resultado más interesante si se confirma.
3. ¿El coste en ms/img compensa? 96 × 320 es 13 veces más píxeles que 24 × 94.

## Después

Esta ablación entrena solo con etiquetas silver, que es donde se aísla el efecto de
la resolución. Una vez elegida la ganadora hay que rehacer el ajuste fino a esa
resolución (`finetune.py`) antes de actualizar las cifras del manuscrito, porque
todas las tablas de la sección 4 reportan el modelo afinado.
