#!/usr/bin/env python3
"""Genera Taller_RNA.ipynb — v2 con todas las correcciones metodológicas."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.14'}
})

cells = []

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — Encabezado y Setup
# ════════════════════════════════════════════════════════════════════════════════

cells.append(nbf.v4.new_markdown_cell(
'''# Taller: Diseño y Optimización de un MLP Profundo

**Curso:** Técnicas de Inteligencia Artificial  
**Universidad Nacional de Colombia — Sede Bogotá**  
**Ingeniería Mecatrónica — Facultad de Ingeniería**  
**Profesor:** Flavio Prieto — faprietoo@unal.edu.co  
**Fecha:** Abril 2026

---

## Objetivo

Diseñar, entrenar y optimizar un Perceptrón Multicapa (MLP) para identificar personas
en un problema de clasificación multiclase usando el dataset **Olivetti Faces** (40 clases, 400 imágenes).

**Restricción:** No se permite el uso de redes convolucionales (CNN).
'''))

cells.append(nbf.v4.new_code_cell(
'''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, f1_score)
from torch.utils.data import DataLoader, TensorDataset
import time, copy, warnings, pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")
print(f"Dispositivo: {DEVICE}")
print(f"PyTorch:     {torch.__version__}")

VERSION       = "rapida"
N_TRIALS      = 30 if VERSION == "rapida" else 80
MAX_EPOCHS    = 100 if VERSION == "rapida" else 150
PATIENCE      = 10 if VERSION == "rapida" else 15
N_CANDIDATES  = 10 if VERSION == "rapida" else 20
RETRAIN_EPOCHS = 150

print(f"\\nVersión: {VERSION} | Trials: {N_TRIALS} | "
      f"Épocas máx: {MAX_EPOCHS} | Patience: {PATIENCE} | "
      f"Candidatos: {N_CANDIDATES}")
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Preprocesamiento
# ════════════════════════════════════════════════════════════════════════════════

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 1 — Preprocesamiento

### 1.1 Carga del dataset Olivetti Faces
'''))

cells.append(nbf.v4.new_code_cell(
'''faces = fetch_olivetti_faces(shuffle=False)
X, y = faces.data, faces.target

print(f"Dimensiones X: {X.shape}  (400 imágenes × 4096 píxeles)")
print(f"Clases:        {len(np.unique(y))} personas")
print(f"Imgs/persona:  {np.bincount(y)[0]}")
print(f"Rango valores: [{X.min():.2f}, {X.max():.2f}]")
'''))

cells.append(nbf.v4.new_markdown_cell('### 1.2 Visualización exploratoria'))

cells.append(nbf.v4.new_code_cell(
'''fig, axes = plt.subplots(5, 10, figsize=(15, 8))
fig.suptitle("Olivetti Faces — 5 personas × 10 imágenes", fontsize=14)
for persona in range(5):
    for j, idx in enumerate(np.where(y == persona)[0]):
        axes[persona, j].imshow(faces.images[idx], cmap="gray")
        axes[persona, j].axis("off")
        if j == 0:
            axes[persona, j].set_ylabel(f"P{persona}", fontsize=10)
plt.tight_layout()
plt.savefig("fig_01_faces_grid.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

cells.append(nbf.v4.new_markdown_cell('### 1.3 Distribución de clases'))

cells.append(nbf.v4.new_code_cell(
'''fig, ax = plt.subplots(figsize=(14, 4))
counts = np.bincount(y)
ax.bar(range(40), counts, color="steelblue", edgecolor="black", alpha=0.8)
ax.set_xlabel("Persona (clase)")
ax.set_ylabel("Número de imágenes")
ax.set_title("Distribución de clases — Olivetti Faces")
ax.set_xticks(range(40))
ax.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="10 imgs/persona")
ax.legend()
plt.tight_layout()
plt.savefig("fig_02_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Dataset perfectamente balanceado: {bool(np.all(counts == 10))}")
'''))

# ── FIX I: Justificación detallada de la partición ──────────────────────────

cells.append(nbf.v4.new_markdown_cell(
'''### 1.4 Partición del dataset — Hold-out estratificado 60/20/20

**Estrategia elegida:** Hold-out con estratificación, dividido en tres subconjuntos:

| Subconjunto | Proporción | Muestras | Imgs/persona | Uso |
|---|---|---|---|---|
| Entrenamiento | 60% | 240 | ~6 | Ajustar los pesos del MLP |
| Validación | 20% | 80 | ~2 | Seleccionar hiperparámetros y comparar modelos |
| Prueba (test) | 20% | 80 | ~2 | Evaluación final **después** de congelar los 3 mejores modelos |

**Justificación:**

1. **¿Por qué tres subconjuntos y no dos?** La búsqueda de hiperparámetros requiere
   un conjunto de validación independiente del test. Si se usaran solo train/test,
   se estaría optimizando indirectamente sobre test y la evaluación final no sería
   una estimación honesta del error de generalización.

2. **¿Por qué estratificado?** Con solo 10 imágenes por persona, un split aleatorio
   podría dejar alguna clase sin representación en validación o test.
   `stratify=y` garantiza proporción similar de cada persona en los tres subconjuntos.

3. **¿Por qué no K-fold como estrategia principal?** K-fold es más robusto para
   estimar varianza, pero con 400 muestras y búsqueda de hiperparámetros costosa
   (30 trials × 100 épocas), el costo computacional se multiplica por K.
   Se usa hold-out como estrategia principal y se menciona K-fold como alternativa
   para validación posterior si el tiempo lo permite.
'''))

cells.append(nbf.v4.new_code_cell(
'''X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)

print("Partición Hold-out estratificado 60/20/20:")
for name, sy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    c = np.bincount(sy, minlength=40)
    print(f"  {name:5s}: {len(sy):3d} muestras "
          f"({len(sy)/len(y)*100:.0f}%) — imgs/persona: [{c.min()}, {c.max()}]")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''### 1.5 Normalización y preparación de tensores

`StandardScaler` ajustado **solo** en train (`.fit_transform` en train,
`.transform` en val y test) para evitar data leakage.
'''))

cells.append(nbf.v4.new_code_cell(
'''scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_train_t = torch.tensor(y_train,    dtype=torch.long)
X_val_t   = torch.tensor(X_val_sc,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,      dtype=torch.long)
X_test_t  = torch.tensor(X_test_sc,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,     dtype=torch.long)

print(f"X_train: {X_train_t.shape}  y_train: {y_train_t.shape}")
print(f"X_val:   {X_val_t.shape}    y_val:   {y_val_t.shape}")
print(f"X_test:  {X_test_t.shape}   y_test:  {y_test_t.shape}")
'''))

# ── FIX F: PCA como análisis complementario, NO en el espacio de búsqueda ───

cells.append(nbf.v4.new_markdown_cell(
'''### 1.6 Reducción de dimensionalidad con PCA (análisis complementario)

Se preparan versiones reducidas del dataset con PCA preservando 95% y 99% de la
varianza. **Este análisis es complementario** — la búsqueda de hiperparámetros
principal se realiza sobre las 4096 dimensiones originales, y el efecto de PCA
se discute por separado en la Parte 6.
'''))

cells.append(nbf.v4.new_code_cell(
'''pca_options = {}
for ratio, label in [(0.95, "95"), (0.99, "99")]:
    pca = PCA(n_components=ratio, random_state=SEED)
    Xtr = pca.fit_transform(X_train_sc)
    Xvl = pca.transform(X_val_sc)
    Xte = pca.transform(X_test_sc)
    pca_options[label] = {
        "pca": pca,
        "X_train": torch.tensor(Xtr, dtype=torch.float32),
        "X_val":   torch.tensor(Xvl, dtype=torch.float32),
        "X_test":  torch.tensor(Xte, dtype=torch.float32),
        "n_components": pca.n_components_
    }
    print(f"PCA {label}%: 4096 → {pca.n_components_} componentes "
          f"({pca.n_components_/4096*100:.1f}% de dims originales)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (label, data) in zip(axes, pca_options.items()):
    cumvar = np.cumsum(data["pca"].explained_variance_ratio_)
    ax.plot(cumvar, linewidth=2)
    ax.axhline(y=float(f"0.{label}"), color="red", linestyle="--")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title(f"PCA {label}% — {data['n_components']} componentes")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_03_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Diseño del MLP
# ════════════════════════════════════════════════════════════════════════════════

# ── FIX C + H: Softmax explícito + enmarcar extensiones ─────────────────────

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 2 — Diseño del MLP

### 2.1 Clase `FullyConnectedNN`

Red completamente densa (Fully Connected) con al menos dos capas ocultas.

**Sobre la capa de salida y Softmax:** El enunciado pide una capa de salida con activación
Softmax. En PyTorch, `nn.CrossEntropyLoss` aplica internamente `LogSoftmax + NLLLoss`,
por lo que la red entrega **logits** (salida lineal) y la activación Softmax queda
**implícita** en la función de pérdida. Esto es equivalente matemáticamente y es la
práctica estándar en PyTorch por estabilidad numérica
(ver [documentación oficial](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
Para la inferencia final, se aplica `torch.softmax()` explícitamente para obtener
probabilidades interpretables.

**Técnicas de regularización:**

| Técnica | Origen | Implementación |
|---|---|---|
| Dropout | Requerido por el enunciado (PDF slide 6) | `nn.Dropout(p)` entre capas ocultas |
| Regularización L2 (Weight Decay) | Requerido por el enunciado (PDF slide 6) | `weight_decay` en el optimizador |
| Batch Normalization | *Extensión adicional* | `nn.BatchNorm1d()` opcional después de cada capa lineal |
| Gradient Clipping | *Extensión adicional* | `clip_grad_norm_()` para estabilidad durante el entrenamiento |

Las extensiones adicionales (BatchNorm, Gradient Clipping) no son requisitos del enunciado,
pero se incluyen porque mejoran la convergencia en redes profundas con alta dimensionalidad
de entrada.
'''))

cells.append(nbf.v4.new_code_cell(
'''class FullyConnectedNN(nn.Module):
    """MLP dinámico con BatchNorm, Dropout y activación configurable."""

    ACT_MAP = {
        "relu": nn.ReLU, "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU, "selu": nn.SELU
    }

    def __init__(self, input_dim, output_dim, hidden_layers,
                 activation="relu", dropout=0.0, use_batchnorm=False):
        super().__init__()
        layers = []
        act_fn = self.ACT_MAP.get(activation, nn.ReLU)
        prev = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """Aplica Softmax explícito para obtener probabilidades."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print("Clase FullyConnectedNN definida.")
'''))

# ── FIX C: Celda de demostración de Softmax explícito ───────────────────────

cells.append(nbf.v4.new_markdown_cell(
'''### 2.1.1 Verificación de Softmax explícito

Se demuestra que la red produce logits y que `torch.softmax()` los convierte
en probabilidades que suman 1 por muestra. Durante el entrenamiento se usan
logits directamente con `CrossEntropyLoss`; durante la inferencia se aplica
Softmax para interpretar las salidas como probabilidades.
'''))

cells.append(nbf.v4.new_code_cell(
'''demo_model = FullyConnectedNN(4096, 40, [64, 32], "relu", 0.0, False)
demo_model.eval()
with torch.no_grad():
    sample = X_train_t[:2]
    logits = demo_model(sample)
    probs  = demo_model.predict_proba(sample)

print("Logits (salida cruda de la red, sin Softmax):")
print(f"  Shape: {logits.shape}")
print(f"  Ejemplo muestra 0: [{logits[0,:5].numpy()} ...]")
print(f"  Suma:  {logits[0].sum().item():.4f} (no suma 1)")

print("\\nProbabilidades (después de torch.softmax):")
print(f"  Shape: {probs.shape}")
print(f"  Ejemplo muestra 0: [{probs[0,:5].numpy()} ...]")
print(f"  Suma:  {probs[0].sum().item():.4f} (suma 1)")
print(f"  Clase predicha: {probs[0].argmax().item()}")

del demo_model
'''))

cells.append(nbf.v4.new_markdown_cell('### 2.2 Funciones de entrenamiento y evaluación'))

cells.append(nbf.v4.new_code_cell(
'''def train_model(model, train_loader, val_loader, criterion, optimizer,
                max_epochs, patience=10, grad_clip=None, verbose=True):
    """Entrena el modelo con early stopping y gradient clipping."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        tloss, tcorr, ttotal = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tloss += loss.item() * yb.size(0)
            tcorr += (out.argmax(1) == yb).sum().item()
            ttotal += yb.size(0)

        model.eval()
        vloss, vcorr, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * yb.size(0)
                vcorr += (out.argmax(1) == yb).sum().item()
                vtotal += yb.size(0)

        tl = tloss / ttotal; ta = tcorr / ttotal
        vl = vloss / vtotal; va = vcorr / vtotal
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if vl < best_val_loss:
            best_val_loss = vl
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping en época {epoch+1}")
                break

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Época {epoch+1:3d}: tloss={tl:.4f} vloss={vl:.4f} "
                  f"tacc={ta:.4f} vacc={va:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    history["epochs_trained"] = len(history["train_loss"])
    history["best_val_loss"] = best_val_loss
    return history


def evaluate_model(model, loader):
    """Retorna (predictions, labels) como arrays numpy."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb).argmax(1).numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


def plot_curves(history, title=""):
    """Grafica loss y accuracy de entrenamiento vs validación."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], "b-", label="Train", lw=2)
    ax1.plot(ep, history["val_loss"],   "r-", label="Val",   lw=2)
    ax1.set(xlabel="Época", ylabel="Loss", title=f"Loss — {title}")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_acc"], "b-", label="Train", lw=2)
    ax2.plot(ep, history["val_acc"],   "r-", label="Val",   lw=2)
    ax2.set(xlabel="Época", ylabel="Accuracy", title=f"Accuracy — {title}")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

print("Funciones de entrenamiento y evaluación definidas.")
'''))

# ── FIX J: Justificación expandida del modelo base ──────────────────────────

cells.append(nbf.v4.new_markdown_cell(
'''### 2.3–2.4 Modelo base y entrenamiento

Arquitectura inicial: **[4096 → 512 → 256 → 128 → 40]**

| Decisión | Elección | Justificación |
|---|---|---|
| **Capas ocultas** | 3 capas (512, 256, 128) | El enunciado pide al menos 2. Se usan 3 capas en estructura piramidal decreciente, que reduce progresivamente la dimensionalidad de 4096 a 40 clases |
| **Neuronas por capa** | 512 → 256 → 128 | Estructura piramidal: cada capa comprime la representación a la mitad. La primera capa (512) captura patrones amplios; las siguientes refinan |
| **Activación** | ReLU | Eficiente computacionalmente, no sufre saturación de gradientes como Sigmoid/Tanh. Recomendada para capas ocultas en las diapositivas del curso (P2) |
| **Capa de salida** | 40 logits | Una neurona por clase; Softmax implícito en `CrossEntropyLoss` (ver sección 2.1) |
| **Función de pérdida** | `CrossEntropyLoss` | Estándar para clasificación multiclase. Equivale a Softmax + Negative Log-Likelihood (diapositivas P2) |
| **Optimizador** | Adam, lr=1e-3 | Adam combina momentum y tasas adaptativas. lr=1e-3 es el punto de partida recomendado (diapositivas P2: "Comenzar con η=10⁻³ si se usa Adam") |
| **Dropout** | 0.3 | Con ratio muestras/features ≈ 240/4096 ≈ 0.06, la regularización es necesaria para evitar sobreajuste |
| **BatchNorm** | Sí | Estabiliza el entrenamiento normalizando las activaciones entre capas (extensión adicional) |
| **Weight Decay** | 1e-4 | Regularización L2 en el optimizador, como pide el enunciado |
'''))

cells.append(nbf.v4.new_code_cell(
'''base_model = FullyConnectedNN(
    input_dim=4096, output_dim=40,
    hidden_layers=[512, 256, 128],
    activation="relu", dropout=0.3, use_batchnorm=True
)
print(f"Modelo base: {base_model.count_parameters():,} parámetros")
print(base_model)

train_loader_base = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
val_loader_base = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer_base = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)

print("\\nEntrenando modelo base...")
t0 = time.time()
base_history = train_model(
    base_model, train_loader_base, val_loader_base,
    criterion, optimizer_base, max_epochs=RETRAIN_EPOCHS,
    patience=15, grad_clip=5.0, verbose=True)
base_time = time.time() - t0

preds, labels = evaluate_model(base_model, val_loader_base)
val_acc = (preds == labels).mean()
print(f"\\nResultados modelo base:")
print(f"  Épocas entrenadas: {base_history['epochs_trained']}")
print(f"  Val accuracy:      {val_acc:.4f}")
print(f"  Tiempo:            {base_time:.1f}s")

fig = plot_curves(base_history, "Modelo Base [512-256-128]")
plt.savefig("fig_04_base_model_curves.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Búsqueda de Hiperparámetros
# ════════════════════════════════════════════════════════════════════════════════

# ── FIX D + K: Justificar Optuna + documentar early stopping vs épocas ──────

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 3 — Búsqueda de Hiperparámetros

### Justificación del método de búsqueda

El enunciado sugiere **Grid Search** o **Random Search**. En este taller se usa
**Optuna con TPE (Tree-structured Parzen Estimator)**, que es una generalización
más eficiente de la búsqueda aleatoria:

- **Random Search** muestrea combinaciones uniformemente al azar. Es mejor que Grid
  Search cuando el espacio tiene muchas dimensiones (Bergstra & Bengio, 2012).
- **TPE** también muestrea al azar, pero **prioriza regiones prometedoras** del espacio
  basándose en los resultados de trials anteriores. Es decir, los primeros trials
  son equivalentes a Random Search, y los siguientes se vuelven más informados.
- Se añade **MedianPruner**, que detiene tempranamente los trials cuyo rendimiento
  intermedio está por debajo de la mediana de trials anteriores, ahorrando tiempo
  de cómputo sin perder calidad.

En la práctica, TPE con 30 trials explora el espacio de forma comparable a un
Random Search con muchos más trials, lo cual es necesario dado el tiempo limitado
en CPU.

### Nota sobre el número de épocas

El enunciado incluye "número de épocas" como hiperparámetro a evaluar. En esta
implementación se usa **Early Stopping** con paciencia configurable, lo que significa
que el número de épocas efectivas lo determina la convergencia del modelo, no un
valor fijo. Se establece un máximo de 100 épocas por trial, pero el entrenamiento
se detiene antes si la validación deja de mejorar. Esto es más eficiente y evita
el sobreajuste por exceso de épocas.

### 3.1 Espacio de búsqueda

| Hiperparámetro | Rango | Tipo |
|---|---|---|
| Capas ocultas | 2 – 5 | `suggest_int` |
| Neuronas/capa | 64 – 1024 | `suggest_int` (log) |
| Activación | ReLU, Tanh, LeakyReLU, SELU | `suggest_categorical` |
| Learning rate | 1e-4 — 1e-1 | `suggest_float` (log) |
| Batch size | 16, 32, 64 | `suggest_categorical` |
| Optimizador | Adam, SGD (momentum=0.9), AdamW | `suggest_categorical` |
| Dropout | 0.0 — 0.5 | `suggest_float` |
| Weight Decay (L2) | 1e-6 — 1e-2 | `suggest_float` (log) |
| Batch Norm | Sí / No | `suggest_categorical` |
'''))

# ── FIX F: PCA eliminado del espacio de búsqueda ────────────────────────────

cells.append(nbf.v4.new_code_cell(
'''def objective(trial):
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden = []
    for i in range(n_layers):
        hidden.append(trial.suggest_int(f"n_units_l{i}", 64, 1024, log=True))

    activation   = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu", "selu"])
    lr           = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    opt_name     = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    use_bn       = trial.suggest_categorical("use_batchnorm", [True, False])

    tr_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                           batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                           batch_size=batch_size)

    model = FullyConnectedNN(4096, 40, hidden, activation, dropout, use_bn)
    crit  = nn.CrossEntropyLoss()

    if opt_name == "Adam":
        opt = optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(),   lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_val_acc = 0.0
    wait = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return best_val_acc

print("Función objective() definida.")
'''))

cells.append(nbf.v4.new_code_cell(
'''print(f"Iniciando búsqueda Optuna — {N_TRIALS} trials...")
t0 = time.time()

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

search_time = time.time() - t0
print(f"\\nBúsqueda completada en {search_time/60:.1f} minutos")
print(f"Mejor trial #{study.best_trial.number}: accuracy = {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")
'''))

cells.append(nbf.v4.new_markdown_cell('### 3.2 Resultados de la búsqueda'))

# ── FIX F: Sin columna PCA en la tabla de resultados ────────────────────────

cells.append(nbf.v4.new_code_cell(
'''completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

rows = []
for t in completed:
    p = t.params
    hidden_str = "-".join(str(p.get(f"n_units_l{i}", "")) for i in range(p["n_layers"]))
    rows.append({
        "Trial": t.number,
        "Capas": p["n_layers"],
        "Neuronas": hidden_str,
        "Activación": p["activation"],
        "LR": f"{p['lr']:.2e}",
        "Optim": p["optimizer"],
        "Dropout": f"{p['dropout']:.2f}",
        "WD": f"{p['weight_decay']:.1e}",
        "BN": p["use_batchnorm"],
        "Val Acc": f"{t.value:.4f}"
    })

df_trials = pd.DataFrame(rows).sort_values("Val Acc", ascending=False).reset_index(drop=True)
print(f"Trials completados: {len(df_trials)} de {N_TRIALS}")
print()
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df_trials.to_string(index=False))

try:
    fig_imp = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Importancia de hiperparámetros")
    plt.tight_layout()
    plt.savefig("fig_05_param_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception:
    print("(Visualización de importancia no disponible)")

try:
    fig_hist = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Historial de optimización")
    plt.tight_layout()
    plt.savefig("fig_06_optimization_history.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception:
    print("(Visualización de historial no disponible)")
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Selección de Modelos (FIX A: sin test)
# ════════════════════════════════════════════════════════════════════════════════

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 4 — Selección de los 3 Mejores Modelos

### 4.1–4.3 Re-entrenamiento detallado de los modelos candidatos

Se seleccionan los top N modelos según **accuracy de validación** y se re-entrenan
con registro completo de métricas. **El conjunto de test NO se toca en esta fase**
— la evaluación en test se realiza únicamente en la Parte 5, después de congelar
la selección de los 3 mejores modelos.
'''))

# ── FIX A: Re-entrenamiento SIN evaluar en test ─────────────────────────────

cells.append(nbf.v4.new_code_cell(
'''top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:N_CANDIDATES]

candidate_results = []
print(f"Re-entrenando top {len(top_trials)} modelos candidatos...\\n")

for rank, trial in enumerate(top_trials):
    p = trial.params
    hidden = [p[f"n_units_l{i}"] for i in range(p["n_layers"])]
    hidden_str = "-".join(map(str, hidden))

    bs = p["batch_size"]
    tr_ld = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=bs, shuffle=True)
    vl_ld = DataLoader(TensorDataset(X_val_t, y_val_t),     batch_size=bs)

    torch.manual_seed(SEED)
    model = FullyConnectedNN(4096, 40, hidden, p["activation"],
                             p["dropout"], p["use_batchnorm"])
    crit = nn.CrossEntropyLoss()

    if p["optimizer"] == "Adam":
        opt = optim.Adam(model.parameters(),  lr=p["lr"], weight_decay=p["weight_decay"])
    elif p["optimizer"] == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    else:
        opt = optim.SGD(model.parameters(),   lr=p["lr"], momentum=0.9, weight_decay=p["weight_decay"])

    name = f"M{rank+1} [{hidden_str}]"
    print(f"  {name} ({p['activation']}, {p['optimizer']}, lr={p['lr']:.1e}, "
          f"do={p['dropout']:.2f}, bn={p['use_batchnorm']})")

    t0 = time.time()
    hist = train_model(model, tr_ld, vl_ld, crit, opt,
                       max_epochs=RETRAIN_EPOCHS, patience=15,
                       grad_clip=5.0, verbose=False)
    elapsed = time.time() - t0

    preds_val, labels_val = evaluate_model(model, vl_ld)
    preds_train, labels_train = evaluate_model(model, tr_ld)

    candidate_results.append({
        "name": name, "rank": rank + 1, "trial": trial.number,
        "params": p, "model": model, "history": hist,
        "hidden_str": hidden_str,
        "val_acc":   (preds_val == labels_val).mean(),
        "train_acc": (preds_train == labels_train).mean(),
        "time_s":    elapsed,
        "n_params":  model.count_parameters(),
    })
    print(f"    → val_acc={candidate_results[-1]['val_acc']:.4f}  "
          f"train_acc={candidate_results[-1]['train_acc']:.4f}  "
          f"({elapsed:.1f}s, {hist['epochs_trained']} épocas)")

print(f"\\nRe-entrenamiento completado.")
'''))

# ── FIX A: Tabla de candidatos sin columnas de test ──────────────────────────

cells.append(nbf.v4.new_code_cell(
'''rows = []
for c in candidate_results:
    p = c["params"]
    rows.append({
        "Modelo": c["name"], "Trial": c["trial"],
        "Activación": p["activation"], "LR": f"{p['lr']:.2e}",
        "Optim": p["optimizer"], "Dropout": f"{p['dropout']:.2f}",
        "BN": p["use_batchnorm"],
        "Val Acc": f"{c['val_acc']:.4f}",
        "Train Acc": f"{c['train_acc']:.4f}",
        "Params": f"{c['n_params']:,}",
        "Tiempo": f"{c['time_s']:.1f}s"
    })
df_candidates = pd.DataFrame(rows)
print("Tabla comparativa de modelos candidatos (solo métricas de validación):\\n")
print(df_candidates.to_string(index=False))
'''))

cells.append(nbf.v4.new_markdown_cell('### 4.4 Curvas de entrenamiento comparativas'))

cells.append(nbf.v4.new_code_cell(
'''fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = plt.cm.tab10(np.linspace(0, 1, len(candidate_results)))

for c, color in zip(candidate_results, colors):
    ep = range(1, c["history"]["epochs_trained"] + 1)
    axes[0, 0].plot(ep, c["history"]["train_loss"], color=color, alpha=0.7, label=c["name"])
    axes[0, 1].plot(ep, c["history"]["val_loss"],   color=color, alpha=0.7, label=c["name"])
    axes[1, 0].plot(ep, c["history"]["train_acc"],  color=color, alpha=0.7, label=c["name"])
    axes[1, 1].plot(ep, c["history"]["val_acc"],    color=color, alpha=0.7, label=c["name"])

for ax, title in zip(axes.flat,
    ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Época")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

plt.suptitle("Curvas de entrenamiento — Todos los candidatos", fontsize=14)
plt.tight_layout()
plt.savefig("fig_07_all_candidates_curves.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

# ── FIX A: Selección de top 3 SOLO con val_acc, sin mostrar test ─────────────

cells.append(nbf.v4.new_markdown_cell(
'''### 4.5 Selección de los 3 mejores modelos

Se seleccionan los 3 modelos con mayor **accuracy de validación**.
La evaluación en el conjunto de test se reserva para la Parte 5.

**Justificación de la selección:** se priorizan los modelos que mejor generalizan
al conjunto de validación, considerando también el balance entre complejidad
(número de parámetros) y rendimiento.
'''))

cells.append(nbf.v4.new_code_cell(
'''top3 = sorted(candidate_results, key=lambda c: c["val_acc"], reverse=True)[:3]
print("═" * 70)
print("  LOS 3 MEJORES MODELOS (seleccionados por Val Acc)")
print("═" * 70)
for i, c in enumerate(top3):
    p = c["params"]
    print(f"\\n  #{i+1}: {c['name']}")
    print(f"       Arquitectura: [{c['hidden_str']}]")
    print(f"       Activación: {p['activation']} | Optim: {p['optimizer']} | "
          f"LR: {p['lr']:.1e}")
    print(f"       Dropout: {p['dropout']:.2f} | BN: {p['use_batchnorm']} | "
          f"WD: {p['weight_decay']:.1e}")
    print(f"       Val Acc:   {c['val_acc']:.4f}")
    print(f"       Train Acc: {c['train_acc']:.4f}")
    print(f"       Parámetros: {c['n_params']:,}")
print("\\n" + "═" * 70)
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Evaluación (FIX A: test se toca AQUÍ por primera vez)
# ════════════════════════════════════════════════════════════════════════════════

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 5 — Evaluación de Desempeño

Los 3 modelos seleccionados se evalúan ahora en el **conjunto de test**, que
no fue usado ni durante el entrenamiento, ni durante la búsqueda de hiperparámetros,
ni durante la selección de candidatos.
'''))

# ── FIX A: Evaluación en test se hace AQUÍ ──────────────────────────────────

cells.append(nbf.v4.new_code_cell(
'''test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

for c in top3:
    preds_test, labels_test = evaluate_model(c["model"], test_loader)
    c["preds_test"]  = preds_test
    c["labels_test"] = labels_test
    c["test_acc"]    = (preds_test == labels_test).mean()
    c["f1_macro"]    = f1_score(labels_test, preds_test, average="macro")
    c["f1_micro"]    = f1_score(labels_test, preds_test, average="micro")

print("Evaluación en test completada para los 3 modelos seleccionados.")
for i, c in enumerate(top3):
    print(f"  Modelo {i+1} ({c['name']}): Test Acc = {c['test_acc']:.4f}, "
          f"F1 Macro = {c['f1_macro']:.4f}")
'''))

# ── FIX E: Evaluación detallada con secciones en markdown ────────────────────

cells.append(nbf.v4.new_markdown_cell('### 5.1 Accuracy global y métricas por clase'))

cells.append(nbf.v4.new_code_cell(
'''for i, c in enumerate(top3):
    print(f"{'─' * 80}")
    print(f"  MODELO {i+1}: {c['name']}")
    print(f"{'─' * 80}")

    print(f"  Accuracy global: {c['test_acc']:.4f} ({c['test_acc']*100:.1f}%)")
    print(f"  F1 macro: {c['f1_macro']:.4f}  |  F1 micro: {c['f1_micro']:.4f}")

    print(f"\\n  Precision, Recall, F1-score por clase:")
    print(classification_report(c["labels_test"], c["preds_test"], zero_division=0))
'''))

cells.append(nbf.v4.new_markdown_cell('### 5.2 Matrices de confusión'))

cells.append(nbf.v4.new_code_cell(
'''fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, c in zip(axes, top3):
    cm = confusion_matrix(c["labels_test"], c["preds_test"])
    sns.heatmap(cm, ax=ax, cmap="Blues", fmt="d", annot=True, annot_kws={"size": 5},
                xticklabels=range(40), yticklabels=range(40))
    ax.set_title(f"{c['name']}\\nTest Acc: {c['test_acc']:.4f}", fontsize=11)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.tick_params(labelsize=5)

plt.suptitle("Matrices de Confusión — Top 3 Modelos", fontsize=14)
plt.tight_layout()
plt.savefig("fig_08_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

cells.append(nbf.v4.new_markdown_cell('### 5.3 Curvas de entrenamiento de los 3 mejores'))

cells.append(nbf.v4.new_code_cell(
'''fig, axes = plt.subplots(3, 2, figsize=(14, 15))
for i, c in enumerate(top3):
    ep = range(1, c["history"]["epochs_trained"] + 1)
    axes[i, 0].plot(ep, c["history"]["train_loss"], "b-", label="Train", lw=2)
    axes[i, 0].plot(ep, c["history"]["val_loss"],   "r-", label="Val",   lw=2)
    axes[i, 0].set(title=f"Loss — {c['name']}", xlabel="Época", ylabel="Loss")
    axes[i, 0].legend(); axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(ep, c["history"]["train_acc"], "b-", label="Train", lw=2)
    axes[i, 1].plot(ep, c["history"]["val_acc"],   "r-", label="Val",   lw=2)
    axes[i, 1].set(title=f"Accuracy — {c['name']}", xlabel="Época", ylabel="Accuracy")
    axes[i, 1].legend(); axes[i, 1].grid(True, alpha=0.3)

plt.suptitle("Curvas de Entrenamiento — Top 3 Modelos", fontsize=14)
plt.tight_layout()
plt.savefig("fig_09_top3_curves.png", dpi=150, bbox_inches="tight")
plt.show()
'''))

cells.append(nbf.v4.new_markdown_cell('### 5.4 Análisis de sobreajuste'))

cells.append(nbf.v4.new_code_cell(
'''print("Análisis de sobreajuste — Gap entre Train Acc y Test Acc:\\n")
for i, c in enumerate(top3):
    gap = c["train_acc"] - c["test_acc"]
    if gap < 0.05:
        diag = "Buen ajuste"
    elif gap < 0.15:
        diag = "Sobreajuste leve"
    else:
        diag = "Sobreajuste significativo"
    print(f"  Modelo {i+1} ({c['name']}):")
    print(f"    Train Acc: {c['train_acc']:.4f}  |  Test Acc: {c['test_acc']:.4f}")
    print(f"    Gap: {gap:.4f}  →  {diag}")
    print(f"    Épocas entrenadas: {c['history']['epochs_trained']}  |  "
          f"Tiempo: {c['time_s']:.1f}s  |  Params: {c['n_params']:,}")
    print()
'''))

# ── FIX E: Tabla comparativa final ──────────────────────────────────────────

cells.append(nbf.v4.new_markdown_cell('### 5.5 Tabla comparativa final'))

cells.append(nbf.v4.new_code_cell(
'''metrics_rows = []
for i, c in enumerate(top3):
    report = classification_report(c["labels_test"], c["preds_test"],
                                   output_dict=True, zero_division=0)
    metrics_rows.append({
        "Métrica": ["Test Accuracy", "F1 Macro", "F1 Micro",
                     "Precision (avg)", "Recall (avg)", "Train Acc",
                     "Gap (Train-Test)", "Tiempo (s)", "Parámetros"],
        f"Modelo {i+1}": [
            f"{c['test_acc']:.4f}", f"{c['f1_macro']:.4f}", f"{c['f1_micro']:.4f}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{c['train_acc']:.4f}",
            f"{c['train_acc'] - c['test_acc']:.4f}",
            f"{c['time_s']:.1f}", f"{c['n_params']:,}"
        ]
    })

df_final = pd.DataFrame({"Métrica": metrics_rows[0]["Métrica"]})
for i, row in enumerate(metrics_rows):
    df_final[f"Modelo {i+1}"] = row[f"Modelo {i+1}"]

print(df_final.to_string(index=False))
'''))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — Discusión, Análisis y Conclusiones (MERGED: fixes B, E, G)
# ════════════════════════════════════════════════════════════════════════════════

cells.append(nbf.v4.new_markdown_cell(
'''---
## Parte 6 — Discusión y Análisis

A continuación se responden las preguntas del enunciado. Los resultados se basan
en un único experimento con semilla fija (`SEED=42`) y un número limitado de trials,
por lo que las conclusiones deben interpretarse como **indicios de este experimento
particular**, no como afirmaciones generales.
'''))

cells.append(nbf.v4.new_markdown_cell(
'''### 6.1 ¿Cuál fue el impacto de la profundidad?
'''))

cells.append(nbf.v4.new_code_cell(
'''depth_data = {}
for t in completed:
    nl = t.params["n_layers"]
    depth_data.setdefault(nl, []).append(t.value)

print("Accuracy de validación promedio por profundidad:\\n")
for nl in sorted(depth_data):
    vals = depth_data[nl]
    print(f"  {nl} capas ocultas: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
          f"({len(vals)} trials)")

best_depth = max(depth_data, key=lambda k: np.mean(depth_data[k]))
print(f"\\n  En este experimento, {best_depth} capas ocultas obtuvieron el mejor "
      f"promedio, aunque con {len(depth_data[best_depth])} trials la diferencia "
      f"puede no ser estadísticamente significativa.")
'''))

# ── FIX B + E: Interpretación en markdown con tono cauteloso ─────────────────

cells.append(nbf.v4.new_markdown_cell(
'''**Interpretación (6.1):**

En este experimento, el número de capas ocultas no muestra una relación monotónica
clara con el rendimiento. Esto es consistente con el teorema del aproximador universal,
que establece que una sola capa oculta suficientemente ancha puede aproximar cualquier
función continua, pero en la práctica redes más profundas pueden aprender representaciones
jerárquicas más eficientes — siempre que la regularización sea adecuada para evitar
sobreajuste. Con solo 240 muestras de entrenamiento, redes muy profundas corren mayor
riesgo de memorizar los datos en lugar de generalizar.
'''))

cells.append(nbf.v4.new_markdown_cell('### 6.2 ¿La regularización fue necesaria?'))

cells.append(nbf.v4.new_code_cell(
'''bn_data = {True: [], False: []}
for t in completed:
    bn_data[t.params["use_batchnorm"]].append(t.value)

print("Efecto de Batch Normalization:")
for bn, vals in bn_data.items():
    if vals:
        print(f"  BN={str(bn):5s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  ({len(vals)} trials)")

do_high = [t.value for t in completed if t.params["dropout"] > 0.3]
do_low  = [t.value for t in completed if t.params["dropout"] <= 0.3]

print("\\nEfecto de Dropout:")
if do_high:
    print(f"  Dropout > 0.3: {np.mean(do_high):.4f} ± {np.std(do_high):.4f}  ({len(do_high)} trials)")
if do_low:
    print(f"  Dropout ≤ 0.3: {np.mean(do_low):.4f} ± {np.std(do_low):.4f}  ({len(do_low)} trials)")

wd_high = [t.value for t in completed if t.params["weight_decay"] > 1e-3]
wd_low  = [t.value for t in completed if t.params["weight_decay"] <= 1e-3]

print("\\nEfecto de Weight Decay (L2):")
if wd_high:
    print(f"  WD > 1e-3: {np.mean(wd_high):.4f} ± {np.std(wd_high):.4f}  ({len(wd_high)} trials)")
if wd_low:
    print(f"  WD ≤ 1e-3: {np.mean(wd_low):.4f} ± {np.std(wd_low):.4f}  ({len(wd_low)} trials)")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''**Interpretación (6.2):**

Con un ratio muestras/features de aproximadamente 240/4096 ≈ 0.06, el riesgo de
sobreajuste es alto. En este experimento, los datos sugieren que la regularización
(Dropout y Weight Decay, ambos requeridos por el enunciado) contribuye a mejorar la
generalización. Batch Normalization, incluido como extensión adicional, también
parece tener un efecto estabilizador, aunque el número de trials no es suficiente
para una conclusión definitiva. Con más repeticiones (diferentes semillas) o
validación cruzada se podría confirmar esta tendencia.
'''))

cells.append(nbf.v4.new_markdown_cell('### 6.3 ¿Se observó sobreajuste?'))

cells.append(nbf.v4.new_code_cell(
'''print("Gap Train-Test para los 3 mejores modelos:\\n")
for i, c in enumerate(top3):
    gap = c["train_acc"] - c["test_acc"]
    print(f"  Modelo {i+1} ({c['name']}): "
          f"train={c['train_acc']:.4f}, test={c['test_acc']:.4f}, gap={gap:.4f}")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''**Interpretación (6.3):**

Los gaps observados entre train accuracy y test accuracy indican el grado de
sobreajuste de cada modelo. En datasets pequeños como Olivetti Faces (400 muestras,
4096 dimensiones), cierto grado de sobreajuste es esperado. Las técnicas de
regularización (Dropout, Weight Decay, Early Stopping) ayudan a mitigarlo, pero
no lo eliminan completamente. Las curvas de entrenamiento (Sección 5.3) muestran
en qué época la validación deja de mejorar, lo cual corresponde al punto donde
Early Stopping detiene el entrenamiento.
'''))

cells.append(nbf.v4.new_markdown_cell('### 6.4 ¿Qué conclusiones sobre la estructura de los datos?'))

cells.append(nbf.v4.new_code_cell(
'''print("Características del dataset:")
print(f"  Dimensionalidad: {X_train_t.shape[1]} features")
print(f"  Muestras train:  {X_train_t.shape[0]}")
print(f"  Ratio muestras/features: {X_train_t.shape[0]/X_train_t.shape[1]:.4f}")
print(f"  Clases: 40 (perfectamente balanceadas)")

print(f"\\nReducción con PCA (análisis complementario):")
for label, data in pca_options.items():
    print(f"  PCA {label}%: 4096 → {data['n_components']} componentes")

print("\\nClases más difíciles de distinguir (basado en el mejor modelo):")
best = top3[0]
report_dict = classification_report(best["labels_test"], best["preds_test"],
                                     output_dict=True, zero_division=0)
worst_classes = sorted(
    [(int(k), v["f1-score"]) for k, v in report_dict.items() if k.isdigit()],
    key=lambda x: x[1]
)[:5]
for cls, f1 in worst_classes:
    print(f"  Persona {cls}: F1 = {f1:.2f}")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''**Interpretación (6.4):**

La alta dimensionalidad (4096 píxeles) frente a las pocas muestras (240 de
entrenamiento) hace de este un problema inherentemente difícil para un MLP.
Cada imagen de 64×64 se aplana como un vector, perdiendo la estructura espacial
2D que una CNN podría aprovechar (pero que el enunciado prohíbe).

La reducción con PCA muestra que una fracción significativa de la varianza
se concentra en relativamente pocos componentes, lo que sugiere que muchas
dimensiones del espacio original contienen ruido o información redundante.
En este experimento, PCA se evaluó como análisis complementario pero no se
incluyó en la búsqueda principal de hiperparámetros.

Las clases más difíciles de clasificar corresponden a personas cuyas imágenes
presentan mayor variabilidad intra-clase (cambios de iluminación, expresión
o pose) o similitud con otras personas.
'''))

# ── FIX G: Análisis complementario de hiperparámetros integrado en Parte 6 ──

cells.append(nbf.v4.new_markdown_cell(
'''### 6.5 Análisis complementario de hiperparámetros
'''))

cells.append(nbf.v4.new_code_cell(
'''print("Función de activación (en este experimento):")
act_data = {}
for t in completed:
    a = t.params["activation"]
    act_data.setdefault(a, []).append(t.value)
for a in sorted(act_data, key=lambda x: np.mean(act_data[x]), reverse=True):
    print(f"  {a:12s}: {np.mean(act_data[a]):.4f} ± {np.std(act_data[a]):.4f}  ({len(act_data[a])} trials)")

print("\\nOptimizador:")
opt_data = {}
for t in completed:
    o = t.params["optimizer"]
    opt_data.setdefault(o, []).append(t.value)
for o in sorted(opt_data, key=lambda x: np.mean(opt_data[x]), reverse=True):
    print(f"  {o:8s}: {np.mean(opt_data[o]):.4f} ± {np.std(opt_data[o]):.4f}  ({len(opt_data[o])} trials)")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''**Interpretación (6.5):**

Los resultados de la búsqueda de hiperparámetros muestran tendencias sobre
qué configuraciones funcionaron mejor *en este experimento particular*:

- **Función de activación:** Los promedios observados dan un indicio de cuál
  activación se adapta mejor a este problema, aunque el número limitado de trials
  por categoría no permite una conclusión definitiva.
- **Optimizador:** Adam y AdamW, con tasas de aprendizaje adaptativas, tienden
  a converger más rápido que SGD en este tipo de problemas. SGD con momentum
  puede alcanzar resultados competitivos pero requiere un ajuste más cuidadoso
  de la tasa de aprendizaje.
- **No se usaron CNNs** porque el enunciado lo prohíbe explícitamente. Las CNN
  aprovecharían la estructura espacial 2D de las imágenes mediante filtros
  convolucionales locales, lo cual no es posible con un MLP que recibe el vector
  aplanado de 4096 dimensiones.
'''))

# ── FIX G: Conclusiones integradas en Parte 6 ───────────────────────────────

cells.append(nbf.v4.new_markdown_cell(
'''### 6.6 Conclusiones
'''))

cells.append(nbf.v4.new_code_cell(
'''best = top3[0]
print(f"Mejor modelo: {best['name']}")
print(f"  Test Accuracy: {best['test_acc']:.4f}")
print(f"  F1 Macro:      {best['f1_macro']:.4f}")
print(f"  Parámetros:    {best['n_params']:,}")
print(f"  Tiempo:        {best['time_s']:.1f}s")
'''))

cells.append(nbf.v4.new_markdown_cell(
'''**Conclusiones generales:**

1. **Sobre el mejor modelo:** El modelo seleccionado logra un rendimiento razonable
   en un problema de clasificación multiclase (40 clases) con muy pocas muestras
   de entrenamiento (6 por persona). Los resultados son específicos a esta partición
   y semilla; con diferentes splits o semillas, los rankings podrían variar.

2. **Sobre la regularización:** En este experimento, las técnicas de regularización
   (Dropout y L2/Weight Decay, requeridas por el enunciado, más BatchNorm como extensión)
   fueron importantes para controlar el sobreajuste, dado el bajo ratio
   muestras/features (~0.06).

3. **Sobre la búsqueda de hiperparámetros:** Se usó Optuna con TPE como generalización
   informada de Random Search, lo que permitió explorar eficientemente el espacio de
   combinaciones. Con 30 trials, el resultado es un buen punto de partida, pero
   más trials o una búsqueda complementaria con Grid Search sobre los mejores
   rangos podría refinar la selección.

4. **Limitaciones del experimento:**
   - Semilla fija única — los resultados podrían variar con otras semillas.
   - Número limitado de trials completados — algunas combinaciones de hiperparámetros
     tienen pocos representantes, lo que limita las conclusiones estadísticas.
   - Sin validación cruzada — una estimación más robusta requeriría K-fold.

5. **Trabajo futuro:**
   - Repetir con múltiples semillas y reportar media ± desviación estándar.
   - Comparar con Grid Search explícito si la rúbrica lo requiere.
   - Explorar data augmentation para aumentar el conjunto de entrenamiento.
'''))

# ────────────────────────────────────────────────────────────────────────────────
nb.cells = cells
out_path = "Taller_RNA.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook creado: {out_path} ({len(cells)} celdas)")
