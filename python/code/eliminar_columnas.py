import os
import pandas as pd

# Carpeta donde están tus archivos CSV
carpeta = "/home/estiberz/UNAL/embedded_systems/MPU6050/python/data"  # <-- cámbiala por el path real

# Recorre todos los archivos en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith(".csv"):
        ruta_archivo = os.path.join(carpeta, archivo)
        print(f"Procesando: {archivo}")
        
        # Leer el CSV
        df = pd.read_csv(ruta_archivo)
        
        # Eliminar columnas si existen
        columnas_a_eliminar = ['t_ms', 'a_total_ms2']
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], errors='ignore')
        
        # Guardar el CSV modificado (sobrescribe)
        df.to_csv(ruta_archivo, index=False)
        
print("Proceso completado. Todos los archivos CSV han sido limpiados.")
