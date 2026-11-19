# Redes Neuronales: TP1, TP2 y TP3 sobre 20 Newsgroups

**Autor:** Ramiro Grin, legajo 65014

Clasificación de texto en 20 categorías (dataset 20 Newsgroups, sin headers, footers ni citas), con representación TF-IDF y modelos en PyTorch.

| TP | Contenido | Secciones del notebook |
|---|---|---|
| TP1 | Regresión softmax: preprocesamiento configurable, búsqueda con Optuna, tiempos de convergencia, normalización de la entrada, repeticiones con semillas | 0 a 9 |
| TP2 | MLP: búsqueda de activación, capas, neuronas y optimizadores (Adam, RMSProp, SGD); barrido de capacidad (overfitting); pruning con y sin; control de activación; comparación con la softmax | 10 a 15 |
| TP3 | Regularización: L2 (Adam y AdamW), L1, L1+L2, Dropout, Max-norm y BatchNorm con comparación controlada; búsqueda conjunta con Optuna; modelo final en test | 16 a 27 |

## Estructura

```
Softmax_MLP_Regularizacion_20Newsgroups_Ramiro_Grin.ipynb   # notebook único con las 3 partes
requirements.txt
README.md
results/      # se genera al ejecutar: tablas de trials, semillas y resúmenes (CSV/JSON)
runs/         # se genera al ejecutar: logs de TensorBoard
mlruns/ o mlflow.db   # se genera al ejecutar: experimentos de MLflow
```

Las tres partes están en un único notebook porque son acumulativas: el TP2 compara contra la softmax del TP1 y el TP3 regulariza el MLP sobredimensionado del TP2. Para que esa dependencia sea explícita y reproducible:

- Todas las semillas están fijas (`SEED = 42` para el split, el sampler de Optuna y los modelos finales; `SEEDS = [0..4]` para las repeticiones). Re-ejecutar da los mismos resultados en el mismo hardware.
- Cada parte **guarda sus resultados en `results/`** y la parte siguiente los **lee de disco**, en lugar de usar variables en memoria. Los números del análisis salen siempre de los archivos de esa corrida.
- Las Partes 2 y 3 empiezan con una celda de setup que verifica las dependencias y recrea los datos si faltan.

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook Softmax_MLP_Regularizacion_20Newsgroups_Ramiro_Grin.ipynb
```

En Colab basta con ejecutar la primera celda (`!pip install ...`). Se recomienda GPU para la Parte 3.

Ejecutar **en orden, de arriba a abajo** (*Run all*). Los parámetros globales están en la celda de setup (sección 0):

| Parámetro | Valor | Significado |
|---|---|---|
| `N_TRIALS` | 40 | Trials de Optuna, iguales para softmax, MLP y MLP regularizado |
| `EPOCHS_SEARCH` / `PATIENCE_SEARCH` | 10 / 3 | Épocas máximas y early stopping por trial |
| `MAX_EPOCHS_FINAL` / `PATIENCE_FINAL` | 50 / 5 | Entrenamientos finales |
| `SEEDS` | 0 a 4 | Repeticiones de los modelos finales |
| `EPOCHS_REG` / `REG_SEEDS` (Parte 3) | 20 / 3 semillas | Mismas épocas y semillas para **todas** las técnicas de regularización |

**Tiempo aproximado:** las Partes 1 y 2 tardan del orden de decenas de minutos en CPU. La Parte 3 (unas 46 configuraciones × 3 semillas × 20 épocas de una red 3×1024, más la búsqueda de Optuna) tarda alrededor de una hora con GPU y bastante más en CPU. Para acortar, bajar `EPOCHS_REG` o `REG_SEEDS` **para todas las configuraciones a la vez**, así la comparación sigue siendo controlada.

## Protocolo de evaluación

- **Split:** el train oficial se divide 80/20 estratificado (`random_state=42`). El 20 % es la validación de **todas** las búsquedas y entrenamientos (Partes 1 a 3).
- **Selección:** early stopping sobre la validación, restaurando los pesos de la mejor época.
- **Test:** el test oficial se usa **solo** para reportar los modelos ya elegidos, con 5 semillas (media ± desvío).
- **Comparaciones:** test de Welch entre semillas para decidir si una diferencia supera al ruido.
- **Normalización de la entrada:** TF-IDF con normalización L2 por documento, ajustado solo con train (justificación y comparación en la sección 3.1 y 7.2).

## Visualización

```bash
tensorboard --logdir runs
mlflow ui
```
