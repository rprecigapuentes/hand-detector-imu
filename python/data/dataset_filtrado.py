import pandas as pd
import os

# === CONFIGURA AQUÍ ===
carpeta = "/home/estiberz/UNAL/embedded_systems/MPU6050/python/data"  # cambia al path donde está tu dataset_unico.csv
entrada = os.path.join(carpeta, "dataset_unico.csv")
salida  = os.path.join(carpeta, "dataset_filtrado.csv")
# ======================

# Leer dataset
df = pd.read_csv(entrada)

# Mostrar conteo antes
print(f"Filas antes: {len(df)}")

# Eliminar filas donde gesture o person estén vacíos o NaN
df_filtrado = df.dropna(subset=["gesture", "person"])
df_filtrado = df_filtrado[df_filtrado["gesture"].astype(str).str.strip() != ""]
df_filtrado = df_filtrado[df_filtrado["person"].astype(str).str.strip() != ""]

# Guardar
df_filtrado.to_csv(salida, index=False)

# Mostrar conteo después
print(f"Filas después: {len(df_filtrado)}")
print(f"✅ Archivo limpio guardado en: {salida}")
