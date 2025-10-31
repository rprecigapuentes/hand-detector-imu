import os
import pandas as pd

# === CONFIGURA AQUÍ ===
carpeta = "/home/estiberz/UNAL/embedded_systems/MPU6050/python/data"  # cambia por el path real
salida = os.path.join(carpeta, "dataset_unico.csv")
# ======================

# Lista para guardar cada dataframe
dataframes = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".csv"):
        ruta = os.path.join(carpeta, archivo)
        try:
            df = pd.read_csv(ruta)
            dataframes.append(df)
            print(f"[OK] Añadido: {archivo}")
        except Exception as e:
            print(f"[SKIP] No se pudo leer {archivo}: {e}")

# Unir todos los DataFrames
if dataframes:
    combinado = pd.concat(dataframes, ignore_index=True)
    combinado.to_csv(salida, index=False)
    print(f"\n✅ Archivo combinado guardado como: {salida}")
else:
    print("\n⚠️ No se encontraron archivos CSV válidos en la carpeta.")
