import os
import pandas as pd

# === CONFIGURA AQUÍ ===
carpeta = "/home/estiberz/UNAL/embedded_systems/MPU6050/python/data"  # pon el path real
# ======================

# Mapeo de personas a IDs ascendentes
person_map = {
    "dani": 1,
    "cami": 2,
    "luxo": 3,
    "stee": 4,
    "ale":  5,
}

# Gestos válidos
valid_gestures = {"like", "dislike", "spider", "yessr", "nada", "presente"}

# Columnas objetivo en el orden deseado
sensor_cols_order = ["ax_ms2", "ay_ms2", "az_ms2", "gx_dps", "gy_dps", "gz_dps"]
drop_if_exists = ["t_ms", "a_total_ms2"]

# Acepta archivos con pinta de CSV aunque tengan extensión rara,
# pero si es un .py real con código, obvio va a fallar al leer.
def es_candidato(nombre):
    lower = nombre.lower()
    return "_" in lower and (lower.endswith(".csv") or lower.endswith(".txt") or lower.endswith(".py"))

for archivo in os.listdir(carpeta):
    if not es_candidato(archivo):
        continue

    ruta = os.path.join(carpeta, archivo)
    base, ext = os.path.splitext(os.path.basename(archivo))
    partes = base.lower().split("_")

    # Se espera al menos: gesto_persona_algo
    if len(partes) < 2:
        print(f"[SKIP] Nombre no parseable: {archivo}")
        continue

    gesto = partes[0]
    persona = partes[1]

    if gesto not in valid_gestures:
        print(f"[SKIP] Gesto inválido '{gesto}' en {archivo}")
        continue
    if persona not in person_map:
        print(f"[SKIP] Persona inválida '{persona}' en {archivo}")
        continue

    try:
        # Intentar leer como CSV con separador por defecto (coma)
        df = pd.read_csv(ruta)
    except Exception as e:
        print(f"[SKIP] No se pudo leer como CSV: {archivo} -> {e}")
        continue

    # Eliminar columnas que sobran si existen
    df = df.drop(columns=[c for c in drop_if_exists if c in df.columns], errors="ignore")

    # Agregar columnas nuevas
    df["gesture"] = gesto
    df["person"]  = person_map[persona]

    # Reordenar columnas, preservando solo las que existan
    cols_presentes = [c for c in sensor_cols_order if c in df.columns]
    final_cols = cols_presentes + ["gesture", "person"]

    # Si faltan columnas esperadas, avisar pero igual guardar
    faltantes = [c for c in sensor_cols_order if c not in df.columns]
    if faltantes:
        print(f"[WARN] En {archivo} faltan columnas: {faltantes}. Se guardará con las disponibles.")

    df = df[final_cols]

    # Guardar sobre el mismo archivo
    df.to_csv(ruta, index=False)
    print(f"[OK] Etiquetado y guardado: {archivo}")
