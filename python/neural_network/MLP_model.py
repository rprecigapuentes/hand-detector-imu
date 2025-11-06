# markdown
# MLP para clasificación de gestos con IMU (split leave-one-subject-out: person 5)
# Requisitos: pandas, numpy, scikit-learn, joblib, matplotlib (opcional para la matriz de confusión)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import itertools
import joblib

# -----------------------------
# 1) Cargar datos
# -----------------------------
CSV_PATH = "dataset_labeled.csv"
df = pd.read_csv(CSV_PATH)

# Columnas esperadas
feature_cols = ["ax_ms2","ay_ms2","az_ms2","gx_dps","gy_dps","gz_dps"]
target_col = "gesture"
person_col = "person"

# Validaciones rápidas
missing_cols = set(feature_cols + [target_col, person_col]) - set(df.columns)
if missing_cols:
    raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

# Asegurar tipos razonables
df[person_col] = df[person_col].astype(int)
# Si hay espacios o mayúsculas raras en 'gesture', normalizamos un poco (opcional)
df[target_col] = df[target_col].astype(str).str.strip().str.lower()

# -----------------------------
# 2) Split por sujeto
# -----------------------------
train_persons = {1,2,3,4}
test_person = {5}

train_df = df[df[person_col].isin(train_persons)].copy()
test_df  = df[df[person_col].isin(test_person)].copy()

X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values

# Codificar labels
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].values)
y_test  = le.transform(test_df[target_col].values)  # asume que test no trae clases nuevas

# Para referencia humana
label_mapping = {cls: int(le.transform([cls])[0]) for cls in le.classes_}
print("Clases y codificación:", label_mapping)

# -----------------------------
# 3) Pipeline: StandardScaler + MLP
# -----------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,              # L2
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=100,            # ~100 'épocas' (iteraciones scikit)
    early_stopping=True,     # usa 10% del train interno para parar temprano
    n_iter_no_change=10,
    random_state=42,
    verbose=False
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", mlp)
])

# -----------------------------
# 4) Entrenar
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# 5) Evaluar en sujeto 5
# -----------------------------
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
#f1m = f1_score(y_test, y_pred, average="macro")

print("\n=== Resultados validación (person == 5) ===")
print(f"Accuracy: {acc:.4f}")
#print(f"F1 macro: {f1m:.4f}\n")

print("Reporte por clase:")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# -----------------------------
# 6) Matriz de confusión
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
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
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()

plot_confusion_matrix(cm, classes=le.classes_, normalize=False, title="Confusión (conteos)")
plt.show()

plot_confusion_matrix(cm, classes=le.classes_, normalize=True, title="Confusión (normalizada)")
plt.show()

# -----------------------------
# 7) Guardar modelo (opcional)
# -----------------------------
# joblib.dump(pipe, "mlp_gestos_pipeline.joblib")
# joblib.dump(le, "label_encoder_gestos.joblib")
# Con esto luego puedes: pipe_loaded = joblib.load(...); le_loaded = joblib.load(...)
# y hacer predicciones en vivo con la misma escala/arquitectura.
