# Taller: Diseño y Optimización de un MLP Profundo

**Curso:** Técnicas de Inteligencia Artificial  
**Profesor:** Flavio Prieto — faprietoo@unal.edu.co  
**Universidad Nacional de Colombia — Sede Bogotá**  
**Ingeniería Mecatrónica — Facultad de Ingeniería**  
**Fecha del taller:** 10 de marzo de 2026  
**Entrega:** 26 de abril de 2026

---

## Objetivo general

Diseñar, entrenar y optimizar un Perceptrón Multicapa (MLP), sin utilizar redes
convolucionales (CNN), para identificar personas en un problema de clasificación
multiclase usando el dataset **Olivetti Faces**.

---

## Dataset: Olivetti Faces

- **Fuente:** `sklearn.datasets.fetch_olivetti_faces()` (descarga automática desde
  el repositorio de scikit-learn; no requiere descarga manual)
- 400 imágenes en escala de grises
- 40 personas distintas, 10 imágenes por persona
- Tamaño: 64 × 64 píxeles → cada imagen se aplana como x ∈ R⁴⁰⁹⁶
- Clasificación multiclase con C = 40 clases
- Distribución perfectamente balanceada

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Framework principal del MLP | **PyTorch** (control total sobre arquitectura, entrenamiento, curvas) |
| Métricas de evaluación | **scikit-learn** (`confusion_matrix`, `classification_report`, `f1_score`) |
| Búsqueda de hiperparámetros | **Optuna** (TPE / Random Search / GridSampler) |
| Reducción de dimensionalidad | **scikit-learn** (`PCA`) |
| Normalización | **scikit-learn** (`StandardScaler`) |
| Visualización | **matplotlib**, **seaborn** |
| Hardware | CPU Ryzen 7 5700U (sin GPU dedicada) |

---

## Código a reutilizar del curso

Todo el código base proviene de los notebooks y PDFs del curso para evitar
preguntas del profesor sobre su origen.

| Fuente | Código disponible | Uso en el taller |
|---|---|---|
| `TIA_202060310_P4.ipynb` | `fetch_olivetti_faces()`, `MLPClassifier`, `train_test_split`, `StandardScaler`, `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report` | Carga del dataset, preprocesamiento, métricas |
| `TIA_20260226_2DOSB.ipynb` | Clase `FullyConnectedNN` dinámica (PyTorch), función `objective()` con Optuna, `GridSampler`, ciclo de entrenamiento/validación | Diseño del MLP, búsqueda de hiperparámetros |
| PDF P2 (Parte Dos) | Teoría de hiperparámetros, funciones de activación, optimizadores, regularización, Optuna | Justificaciones teóricas |
| PDF P3 (Parte Tres) | Backpropagation, cálculo de gradientes | Justificaciones teóricas |
| PDF P4 (Parte Cuatro) | Normalización, métricas, desbalance de datos | Justificaciones en preprocesamiento y evaluación |

---

## Estructura del Notebook

### Sección 0: Encabezado y Setup

- Título, autores, fecha
- Imports globales: `torch`, `sklearn`, `optuna`, `numpy`, `matplotlib`, `seaborn`
- Configuración de semillas (`torch.manual_seed`, `np.random.seed`) para reproducibilidad
- Verificación de dispositivo (CPU)
- Definición de constantes globales (versión rápida 1.5h vs versión completa 4h)

---

### Sección 1: Parte 1 — Preprocesamiento

**1.1. Carga del dataset**
- Usar `fetch_olivetti_faces()` (código adaptado del notebook P4)
- Mostrar dimensiones y propiedades del dataset

**1.2. Visualización exploratoria**
- Grid de caras: mostrar las 10 imágenes de al menos 5 personas distintas
- Visualizar variabilidad intra-clase (iluminación, expresión, pose)

**1.3. Análisis de distribución de clases**
- Gráfico de barras confirmando 10 imágenes por persona
- Verificar que el dataset es perfectamente balanceado

**1.4. Partición del dataset — Estrategia dual**

**Estrategia principal (Hold-out estratificado 60/20/20):**
- 6 imágenes train, 2 validación, 2 test por persona
- Split con `train_test_split` y `stratify=y`
- Justificación: dataset pequeño, necesitamos validación para búsqueda de
  hiperparámetros y test independiente para evaluación final

**Estrategia complementaria (Leave-K-Out por persona):**
- Para cada modelo entrenado, dejar K=2 imágenes por persona para validación
- Entrenar con las 8 restantes
- Rotar cuáles se dejan fuera (similar a K-fold estratificado por persona)
- Promediar los resultados de validación de todas las rotaciones
- Esto da una estimación más robusta del rendimiento con tan pocas muestras
- Se implementa como validación cruzada estratificada (`StratifiedKFold` con
  k=5, que equivale a dejar ~2 imágenes por persona fuera en cada fold)

**1.5. Normalización y preparación de tensores**
- `StandardScaler` ajustado solo en train (`.fit_transform(X_train)`,
  `.transform(X_val)`, `.transform(X_test)`)
- Conversión a tensores PyTorch (`torch.float32` para X, `torch.long` para y)
- Creación de `DataLoader` con `TensorDataset`

**1.6. Reducción de dimensionalidad con PCA (variante)**
- Aplicar PCA preservando 95% y 99% de la varianza
- Comparar número de componentes resultantes vs 4096 originales
- Preparar versiones reducidas del dataset para comparación posterior

---

### Sección 2: Parte 2 — Diseño del MLP

**2.1. Definición de la clase `FullyConnectedNN`**
- Adaptar la clase del notebook 2DOSB para:
  - `input_dim=4096` (o dimensión reducida por PCA)
  - `output_dim=40` clases
  - Número variable de capas ocultas (mínimo 2)
  - Soporte para múltiples funciones de activación (ReLU, Tanh, LeakyReLU, SELU)
  - Dropout configurable por capa
  - Batch Normalization opcional
  - Regularización L2 vía `weight_decay` del optimizador

**2.2. Técnicas de regularización a explorar**

| Técnica | Implementación | Hiperparámetro |
|---|---|---|
| Dropout | `nn.Dropout(p)` entre capas | p ∈ [0.0, 0.5] |
| Regularización L2 (Weight Decay) | `weight_decay` en el optimizador | λ ∈ [1e-5, 1e-2] |
| Regularización L1 | Penalización manual en la loss | λ₁ ∈ [1e-6, 1e-3] |
| Batch Normalization | `nn.BatchNorm1d()` después de cada capa lineal | Sí/No |
| Early Stopping | Monitoreo de val_loss con paciencia | patience ∈ [5, 15] |
| Gradient Clipping | `torch.nn.utils.clip_grad_norm_()` | max_norm ∈ [1.0, 5.0] |

**2.3. Modelo base justificado**
- Arquitectura inicial: [4096 → 512 → 256 → 128 → 40]
- Activación: ReLU (justificación: eficiente, sin saturación, recomendada en
  las diapositivas P2 para capas ocultas)
- Salida: Softmax implícito en `CrossEntropyLoss` de PyTorch
- Pérdida: `CrossEntropyLoss` (entropía cruzada multiclase, recomendada en
  diapositivas P2 slide 67)
- Optimizador: Adam con lr=1e-3 (recomendación de diapositivas P2 slide 140:
  "Comenzar con η=10⁻³ si se usa Adam")
- Justificación teórica de cada decisión citando las diapositivas del curso

**2.4. Entrenamiento del modelo base**
- Ciclo de entrenamiento adaptado del notebook 2DOSB
- Registro de loss y accuracy por época (train y validation)
- Gráficas de curvas de entrenamiento

---

### Sección 3: Parte 3 — Búsqueda de Hiperparámetros

**3.1. Definición del espacio de búsqueda**

| Hiperparámetro | Rango | Tipo |
|---|---|---|
| Número de capas ocultas | 2 – 5 | `suggest_int` |
| Neuronas por capa | 64, 128, 256, 512, 1024 | `suggest_int` |
| Función de activación | ReLU, Tanh, LeakyReLU, SELU | `suggest_categorical` |
| Tasa de aprendizaje | 1e-4 — 1e-1 | `suggest_float` (log) |
| Tamaño de lote | 16, 32, 64 | `suggest_categorical` |
| Optimizador | Adam, SGD (momentum=0.9), AdamW | `suggest_categorical` |
| Dropout | 0.0 — 0.5 | `suggest_float` |
| Weight Decay (L2) | 1e-6 — 1e-2 | `suggest_float` (log) |
| Batch Normalization | Sí / No | `suggest_categorical` |
| Épocas | 50, 100, 150 | `suggest_categorical` |
| Usar PCA | No, 95%, 99% | `suggest_categorical` |

**3.2. Versión rápida (~1.5 horas de entrenamiento total)**
- Optuna con TPE sampler: **30 trials**
- Early stopping con paciencia=10 en cada trial
- Épocas máximas por trial: 100
- Pruning de Optuna (`MedianPruner`) para descartar trials malos temprano
- Selección de **10 modelos candidatos** para la Parte 4

**3.3. Versión completa (~4 horas de entrenamiento total)**
- Optuna con TPE sampler: **80 trials**
- Early stopping con paciencia=15 en cada trial
- Épocas máximas por trial: 150
- Grid Search adicional (`GridSampler`) sobre el subespacio de los mejores
  hiperparámetros encontrados por TPE
- Validación cruzada estratificada (5-fold) para los top 20 modelos
- Selección de **20 modelos candidatos** para la Parte 4

**3.4. Adaptación de la función `objective()` del notebook 2DOSB**
- Modificar `input_dim=4096` (o PCA), `output_dim=40`
- Agregar soporte para Batch Normalization, L1, Gradient Clipping
- Agregar Early Stopping dentro del trial
- Usar Optuna pruning para eficiencia

**3.5. Resultados de la búsqueda**
- Tabla de todos los trials con hiperparámetros y accuracy en validación
- Gráficas de Optuna: importancia de hiperparámetros, historial de optimización
- Selección de **10 modelos candidatos** (versión rápida) o **20 modelos candidatos**
  (versión completa), ordenados por accuracy de validación

---

### Sección 4: Parte 4 — Selección de los 3 Mejores Modelos

**4.1. Tabla comparativa de los modelos candidatos (10 o 20 según versión)**

| Modelo | Capas | Neuronas | Activación | LR | Optim | Dropout | L2 | BN | PCA | Val Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**4.2. Re-entrenamiento detallado de todos los candidatos**
- Entrenar cada modelo registrando métricas completas por época
- Loss y accuracy en train y validation por época

**4.3. Curvas de entrenamiento**
- Para cada modelo: gráfica con loss (train vs val) y accuracy (train vs val)
- Superponer curvas de todos los candidatos para comparación visual

**4.4. Validación cruzada (versión completa, 20 modelos)**
- 5-fold estratificado para estimar varianza del rendimiento
- Media ± desviación estándar del accuracy

**4.5. Selección final**
- Presentar la tabla final con los 10 o 20 modelos y sus métricas
- **El usuario escoge los 3 mejores** de entre las opciones presentadas
- Justificación técnica de por qué los modelos seleccionados son superiores

---

### Sección 5: Parte 5 — Evaluación de Desempeño

Para cada uno de los 3 modelos seleccionados, evaluar en el **conjunto de test**
(nunca visto durante entrenamiento ni búsqueda de hiperparámetros):

**5.1. Accuracy global**

**5.2. Matriz de confusión**
- Código reutilizado del notebook P4: `confusion_matrix()`, `ConfusionMatrixDisplay()`
- Visualización con heatmap (40×40 clases)

**5.3. Precision, Recall y F1-score por clase**
- `classification_report()` del notebook P4
- Identificar clases con peor rendimiento y analizar por qué

**5.4. F1 macro y micro**
- F1 macro: promedio no ponderado de F1 por clase (sensible a clases minoritarias)
- F1 micro: F1 global (equivalente a accuracy en datasets balanceados)

**5.5. Análisis de sobreajuste**
- Gap entre train accuracy y test accuracy
- Comparación de curvas train vs validation loss
- Diagnóstico: ¿underfitting, buen ajuste o overfitting?

**5.6. Tiempo de entrenamiento**
- Tiempo total en segundos/minutos para cada modelo
- Relación costo-beneficio (accuracy vs tiempo)

**5.7. Tabla comparativa final**

| Métrica | Modelo 1 | Modelo 2 | Modelo 3 |
|---|---|---|---|
| Test Accuracy | | | |
| F1 Macro | | | |
| F1 Micro | | | |
| Precision (avg) | | | |
| Recall (avg) | | | |
| Train Acc | | | |
| Gap (Train-Test) | | | |
| Tiempo (s) | | | |
| Parámetros totales | | | |

---

### Sección 6: Parte 6 — Discusión y Análisis

Responder explícitamente cada pregunta del taller:

**6.1. ¿Cuál fue el impacto de la profundidad?**
- Comparar modelos con 2 vs 3 vs 4+ capas ocultas
- Analizar si más capas mejoran o empeoran el rendimiento
- Relación con el teorema del aproximador universal (PDF P2, slide 21)

**6.2. ¿La regularización fue necesaria?**
- Comparar modelos con y sin Dropout
- Comparar modelos con y sin Weight Decay (L2)
- Efecto de Batch Normalization
- Efecto de Early Stopping
- Con 400 muestras y 4096 features, la regularización es probablemente crítica

**6.3. ¿Se observó sobreajuste?**
- Evidencia en las curvas de entrenamiento
- Gap train-test accuracy
- Cuántas épocas fueron necesarias antes de que el modelo comience a sobreajustarse

**6.4. ¿Qué conclusiones pueden extraerse sobre la estructura de los datos?**
- Alta dimensionalidad (4096) vs pocas muestras (400)
- Efecto de PCA: ¿la reducción de dimensionalidad ayuda?
- Qué clases (personas) son más difíciles de distinguir y por qué
- Importancia de la normalización

---

### Sección 7: Preguntas adicionales (internas, no del PDF)

Responder preguntas complementarias que enriquecen el análisis:

1. **¿Por qué no usar CNN si son imágenes?** El taller lo prohíbe explícitamente,
   pero conceptualmente las CNN aprovechan la estructura espacial 2D
2. **¿Cómo afecta PCA al rendimiento?** Comparación directa de modelos con y sin PCA
3. **¿Qué función de activación resultó mejor y por qué?** Análisis basado en
   los resultados experimentales
4. **¿Cuál optimizador fue más efectivo?** Adam vs SGD vs AdamW en este problema
5. **¿Batch Normalization es útil aquí?** Análisis del impacto
6. **¿Qué arquitectura (piramidal decreciente vs uniforme) funciona mejor?**
7. **¿Cuántas épocas son suficientes?** Análisis de convergencia

---

### Sección 8: Conclusiones generales

- Resumen de hallazgos principales
- Mejor modelo y por qué
- Lecciones aprendidas sobre el diseño de MLPs
- Recomendaciones para trabajo futuro

---

## Consideraciones de hardware y tiempo

| Versión | Trials Optuna | Épocas/trial | Modelos candidatos | Tiempo estimado | Validación cruzada |
|---|---|---|---|---|---|
| Rápida | 30 | 100 | 10 | ~1.5 horas | Solo hold-out |
| Completa | 80 | 150 | 20 | ~4 horas | 5-fold para top 20 |

**CPU:** Ryzen 7 5700U (8 cores / 16 threads)
- PyTorch puede aprovechar múltiples cores para operaciones de álgebra lineal
- El batch size pequeño (16-64) es adecuado para CPU
- Optuna Pruning (`MedianPruner`) es esencial para no desperdiciar tiempo en
  configuraciones malas

---

## Flujo de trabajo

```
1. Crear notebook con todas las secciones (markdown + código)
2. Ejecutar Secciones 0-1 (preprocesamiento)
3. Ejecutar Sección 2 (modelo base)
4. Ejecutar Sección 3 (búsqueda — versión rápida o completa según tiempo)
5. Presentar tabla de top 10+ modelos al usuario
6. Usuario escoge los 3 mejores modelos
7. Ejecutar Sección 5 (evaluación detallada de los 3 seleccionados)
8. Completar Secciones 6-8 (discusión y conclusiones)
```

---

## Entregables

- [x] Código reproducible y comentado en un único Jupyter Notebook (tipo informe técnico)
- [ ] Tablas comparativas de hiperparámetros y métricas
- [ ] Gráficas de entrenamiento (loss y accuracy)
- [ ] Conclusiones argumentadas en celdas markdown
- [ ] Video explicando la solución (máximo 15 minutos)
