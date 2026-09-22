# Qué script produce cada tabla y figura

Todas las rutas son relativas a `benchmark/`. Requieren el conjunto de datos
descargado de [Zenodo](https://doi.org/10.5281/zenodo.22021714) y descomprimido.

## Preparación

| Paso | Script | Salida |
|---|---|---|
| Construir el conjunto de reconocimiento | `prepare_rec_dataset.py` | `data/rec_labels_{train,val,test}.tsv` |
| Construir el conjunto de detección | `build_det_dataset.py` | `det_data/` |
| Reconstruir el dataset público sin fuga | `clean_roboflow_dataset.py` | `PlacasPeruanas_clean/` |
| Verificar que no hay fuga entre particiones | `gold_leakage_check.py` | informe por consola |
| Dimensionar Gold-B por potencia | `gold2_power.py` | curva de potencia |
| Barrido de rostros y difuminado | `face_scan.py`, `blur_faces.py` | manifiesto y recortes tratados |

## Entrenamiento

| Paso | Script |
|---|---|
| Destilación con etiquetas *silver* | `ablation_resolucion/run_ablation.py --only 32x100 --epochs 10 --seed N` |
| Ajuste fino, ambas arquitecturas, tres semillas | `ablation_resolucion/finetune_all.py` |
| Validación cruzada agrupada por matrícula | `cv_finetune.py` |
| Exportar el modelo desplegable | `export_lprnet_onnx.py` |

`finetune.py` es el que ejecuta un ajuste fino individual, con los tres papeles
disjuntos: gradiente, selección de época y control de fuga.

## Tablas del artículo

| Tabla | Contenido | Script |
|---|---|---|
| 1 | Especificaciones del sensor | manual |
| 2 | Particionado del corpus | `prepare_rec_dataset.py` |
| 3 | Conjuntos de evaluación | manual |
| 4 | Sistemas comparados | manual |
| 5 | Papeles disjuntos en el ajuste fino | manual |
| 6 | Hiperparámetros | manual |
| 7 | Reconocimiento sobre las dos verdades-terreno | `regenerar_cifras_32x100.py` |
| 8 | Contrastes post-hoc | `regenerar_cifras_32x100.py` |
| 9 | Modelo factorial | `regenerar_cifras_32x100.py` |
| 10 | Por número de pasadas | `regenerar_cifras_32x100.py` |
| 11 | Por cuartil de área, sobre Gold-A | `regenerar_cifras_32x100.py` |
| 12 | Por clase lumínica | `regenerar_cifras_32x100.py` |
| 13 | Efecto del ajuste fino | `regenerar_cifras_32x100.py` |
| 14 | Variabilidad entre semillas | `regenerar_cifras_32x100.py` |
| 15 | Ablación de resolución | `ablation_resolucion/report_ablation.py` |
| 16 | Acuerdo inter-anotador | `consolidate_gold.py` |
| 17 | Comparativa de detectores | `det_benchmark.py` |
| 18 | Brecha entre ópticas | `crossdomain_eval.py` |

La numeración es orientativa y puede desplazarse según la versión del manuscrito;
lo estable es qué script genera cada cifra.

## Figuras

| Figura | Contenido | Script |
|---|---|---|
| 1 | Arquitectura del sistema | `fig_arquitectura_sistema.py` |
| 2 | Tipos de matrícula peruana | `fig_tipos_placa.py` |
| 3 | Arquitecturas de los reconocedores | `fig_arquitecturas.py` |
| 4 | Errores representativos | `fig_errores.py` |
| 5 | Exactitud y perfiles por subgrupo | `fig_perfiles.py` |
| 6 | Post-hoc de caja y bigotes | `fig_posthoc_cajas.py` |

Las Figuras 5 y 6 necesitan que `regenerar_cifras_32x100.py` se haya ejecutado antes,
porque leen `results/cifras_32x100.json` y `results/matriz_gold2_32x100.csv`.

## Orden mínimo para regenerar todo el análisis

```bash
cd benchmark
for s in 0 1 2; do
  python ablation_resolucion/run_ablation.py --only 32x100 --epochs 10 --seed $s
done
python ablation_resolucion/finetune_all.py
python regenerar_cifras_32x100.py
python fig_perfiles.py
python fig_posthoc_cajas.py
```

En una RTX 5060 Ti son unas dos horas, casi todas en el paso de destilación.

## Advertencias

`ablation_resolucion/fix_ckpt_keys.py` existe porque los modelos parametrizados por
resolución dividen la pila convolucional del CRNN y eso cambia los nombres de los
parámetros. Si un checkpoint de la ablación no carga en `models.py`, ese script lo
reindexa y verifica numéricamente que la conversión es equivalente.

Los checkpoints antiguos a 24 × 94 siguen cargando: el colapso adaptativo de altura
que introduce `models.py` es idéntico al `MaxPool` original a esa resolución.
