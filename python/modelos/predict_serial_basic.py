#!/usr/bin/env python3
import sys
import time
import joblib
import serial
import numpy as np

# =============================
# CONFIG
# =============================
MODEL_PATH = "modelo_rf.sav"    # cambia por tu .sav (RF o DT)
PORT = "/dev/ttyUSB0"           # en Raspberry puede ser /dev/ttyUSB0 o /dev/ttyACM0
BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0            # segundos

# Features usados en TU entrenamiento (según notebook)
REQUIRED_FEATURES = ["ax_ms2","ay_ms2","az_ms2","gx_dps","gy_dps","gz_dps"]

def parse_header(line):
    """
    Recibe una línea tipo 't_ms,ax_ms2,ay_ms2,...'
    Devuelve un dict col_name -> index para las columnas necesarias.
    """
    cols = [c.strip() for c in line.split(",")]
    index_map = {}
    for f in REQUIRED_FEATURES:
        if f in cols:
            index_map[f] = cols.index(f)
        else:
            raise ValueError(f"Falta la columna '{f}' en el header serial. Recibí: {cols}")
    return index_map

def parse_row(line, index_map):
    """
    Convierte una línea de datos CSV a vector X ordenado según REQUIRED_FEATURES.
    Ignora columnas extra (por ejemplo t_ms).
    """
    parts = [p.strip() for p in line.split(",")]
    values = []
    for f in REQUIRED_FEATURES:
        try:
            v = float(parts[index_map[f]])
        except (ValueError, IndexError):
            raise ValueError(f"Línea inválida o incompleta para '{f}': {line}")
        values.append(v)
    return np.array(values, dtype=float).reshape(1, -1)

def main():
    # Cargar modelo (puede ser un Pipeline de sklearn; mejor)
    print(f"Cargando modelo: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print("Modelo cargado.")

    # Abrir serial
    print(f"Abriendo serial en {PORT} @ {BAUDRATE}...")
    ser = serial.Serial(PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2)  # pequeño respiro para que el micro empiece a enviar
    print("Serial listo.\nEsperando header...")

    index_map = None

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            # Si llega un header (línea con nombres), lo parseamos o re-parseamos
            if any(name in raw for name in REQUIRED_FEATURES):
                try:
                    index_map = parse_header(raw)
                    print(f"Header detectado. Mapeo columnas: {index_map}")
                except ValueError as e:
                    print(f"[WARN] {e}")
                continue

            if index_map is None:
                # Aún no hemos visto el header: ignorar líneas hasta que llegue
                continue

            # Intentar parsear la fila y predecir
            try:
                X = parse_row(raw, index_map)
                y_pred = clf.predict(X)[0]
                # Si tu modelo soporta predict_proba, puedes descomentar para depurar
                # proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
                print(f"Predicción: {y_pred}")
            except ValueError as e:
                # Línea corrupta o incompleta; la saltamos sin matar el proceso
                print(f"[SKIP] {e}")
                continue

    except KeyboardInterrupt:
        print("\nCerrando...")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Permitir pasar ruta del modelo y puerto por CLI si quieres:
    #   python3 predict_serial_basic.py modelo_rf.sav /dev/ttyACM0
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        PORT = sys.argv[2]
    main()
