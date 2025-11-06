# MLP sin la clase 'dislike' · accuracy, recall y matrices de confusión

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
# 1) Cargar y filtrar datos
# -----------------------------
CSV_PATH = "dataset_labeled.csv"
df = pd.read_csv(CSV_PATH)

feature_cols = ["ax_ms2","ay_ms2","az_ms2","gx_dps","gy_dps","gz_dps"]
target_col = "gesture"
person_col = "person"

df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[person_col] = df[person_col].astype(int)

# Excluir 'nada' y 'dislike'
df = df[df[target_col] != "nada"].copy()
df = df[df[target_col] != "dislike"].copy()

# -----------------------------
# 2) Split por sujeto
# -----------------------------
train_persons = {1,2,3,4}
test_person   = {5}

train_df = df[df[person_col].isin(train_persons)].copy()
test_df  = df[df[person_col].isin(test_person)].copy()

X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values

y_train_raw = train_df[target_col].values
y_test_raw  = test_df[target_col].values

# -----------------------------
# 3) Codificar labels
# -----------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

mask_test_known = np.isin(y_test_raw, le.classes_)
if not np.all(mask_test_known):
    desconocidas = sorted(set(y_test_raw[~mask_test_known]))
    print(f"[Aviso] Se excluyen {np.sum(~mask_test_known)} muestras de test con clases no vistas en train: {desconocidas}")
    X_test   = X_test[mask_test_known]
    y_test_raw = y_test_raw[mask_test_known]

y_test = le.transform(y_test_raw)
classes = le.classes_
print("Clases usadas:", list(classes))

# -----------------------------
# 4) Pipeline y entrenamiento
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
# 5) Evaluación
# -----------------------------
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
recall_macro = recall_score(y_test, y_pred, average="macro")
recall_per_class = recall_score(y_test, y_pred, average=None)

print("\n=== Validación (person == 5, sin 'dislike') ===")
print(f"Accuracy: {acc:.4f}")
print(f"Recall macro: {recall_macro:.4f}")
print("Recall por clase:")
for cls, r in zip(classes, recall_per_class):
    print(f"  - {cls}: {r:.4f}")

# -----------------------------
# 6) Matrices de confusión
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
    plt.ylabel('Etiqueta real')
    plt.xlabel('Predicción')
    plt.tight_layout()

plot_confusion_matrix(cm, classes, normalize=False, title="Matriz de confusión (conteos)")
plt.show()

plot_confusion_matrix(cm, classes, normalize=True, title="Matriz de confusión (normalizada)")
plt.show()

# -----------------------------
# 7) Guardar modelo entrenado
# -----------------------------
MODEL_PATH = "mlp_gestos_final.sav"

payload = {
    "pipeline": pipe,
    "label_encoder": le,
    "feature_cols": feature_cols,
    "classes_": list(classes)
}

joblib.dump(payload, MODEL_PATH)
print(f"\nModelo guardado exitosamente en: {os.path.abspath(MODEL_PATH)}")
