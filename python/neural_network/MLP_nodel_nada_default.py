# MLP con "nada" como predeterminada por baja confianza
# - Filtra 'dislike'
# - Train: person in {1,2,3,4} | Test: person == 5
# - Métricas: accuracy, recall (macro y por clase)
# - Matrices de confusión (conteo y normalizada)
# - Umbral de confianza: si max(proba) < CONF_THR => predicción = "nada"
# - Guarda modelo y metadatos en .sav (joblib)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import joblib
import os

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "dataset_labeled.csv"
FEATURES = ["ax_ms2","ay_ms2","az_ms2","gx_dps","gy_dps","gz_dps"]
TARGET = "gesture"
PERSON = "person"

TRAIN_PERSONS = {1,2,3,4}
TEST_PERSON   = {5}

CONF_THR = 0.60   # si la confianza máxima es menor a este umbral => 'nada'
MODEL_PATH = "mlp_gestos_idle_threshold.sav"

# -----------------------------
# 1) Cargar y preparar datos
# -----------------------------
df = pd.read_csv(CSV_PATH)

# saneo básico
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
df[PERSON] = df[PERSON].astype(int)

# Excluir 'dislike'
#df = df[df[TARGET] != "dislike"].copy()

# Split por sujeto
train_df = df[df[PERSON].isin(TRAIN_PERSONS)].copy()
test_df  = df[df[PERSON].isin(TEST_PERSON)].copy()

X_train = train_df[FEATURES].values
X_test  = test_df[FEATURES].values
y_train_raw = train_df[TARGET].values
y_test_raw  = test_df[TARGET].values

# Codificar etiquetas
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# asegurar que test no tenga clases nuevas
mask_known = np.isin(y_test_raw, le.classes_)
if not np.all(mask_known):
    desconocidas = sorted(set(y_test_raw[~mask_known]))
    print(f"[Aviso] Se excluyen {np.sum(~mask_known)} muestras de test con clases no vistas en train: {desconocidas}")
    X_test = X_test[mask_known]
    y_test_raw = y_test_raw[mask_known]

y_test = le.transform(y_test_raw)
classes = le.classes_
print("Clases usadas:", list(classes))

# índice de 'nada'
if "nada" not in classes:
    raise RuntimeError("La clase 'nada' no está en las clases del entrenamiento. Revisa el CSV/filtrado.")
idle_cls_idx = int(np.where(classes == "nada")[0][0])

# -----------------------------
# 2) Pipeline y entrenamiento
# -----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64,32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    ))
])

pipe.fit(X_train, y_train)

# -----------------------------
# 3) Predicción con regla de baja confianza => 'nada'
# -----------------------------
proba = pipe.predict_proba(X_test)           # [n_samples, n_classes]
y_pred = proba.argmax(axis=1)                # clase de máxima probabilidad
maxp = proba.max(axis=1)                     # confianza

# aplicar umbral
low_conf_mask = maxp < CONF_THR
y_pred[low_conf_mask] = idle_cls_idx

# -----------------------------
# 4) Métricas
# -----------------------------
acc = accuracy_score(y_test, y_pred)
recall_macro = recall_score(y_test, y_pred, average="macro")
recall_per_class = recall_score(y_test, y_pred, average=None)

print("\n=== Validación (person == 5, sin 'dislike', con idle por baja confianza) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Recall macro: {recall_macro:.4f}")
print("Recall por clase:")
for cls, r in zip(classes, recall_per_class):
    print(f"  - {cls}: {r:.4f}")

# -----------------------------
# 5) Matrices de confusión
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión'):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = format(cm[i, j], fmt)
        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if (cm[i, j] > thresh) else "black")
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()

plot_confusion_matrix(cm, classes, normalize=False, title="Matriz de confusión (conteos)")
plt.show()

plot_confusion_matrix(cm, classes, normalize=True, title="Matriz de confusión (normalizada)")
plt.show()

# -----------------------------
# 6) Guardar modelo en .sav (joblib)
# -----------------------------
payload = {
    "pipeline": pipe,
    "label_encoder": le,
    "feature_cols": FEATURES,
    "classes_": list(classes),
    "idle_label": "nada",
    "idle_index": idle_cls_idx,
    "confidence_threshold": float(CONF_THR)
}
joblib.dump(payload, MODEL_PATH)
print(f"\nModelo guardado en: {os.path.abspath(MODEL_PATH)}")

# -----------------------------
# 7) Ejemplo de uso (carga + predicción con umbral)
# -----------------------------
# model = joblib.load(MODEL_PATH)
# proba_live = model["pipeline"].predict_proba(X_new)   # X_new con mismas columnas FEATURES
# y_pred_live = proba_live.argmax(1)
# maxp_live = proba_live.max(1)
# y_pred_live[maxp_live < model["confidence_threshold"]] = model["idle_index"]
# gestos = np.array(model["classes_"])[y_pred_live]
