#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import warnings
import joblib
import numpy as np
from smbus2 import SMBus

# Silencia el warning de sklearn por "feature names"
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning
)

# =============================
# CONFIG
# =============================
I2C_BUS = 1            # GPIO2 (SDA) y GPIO3 (SCL) en Raspberry
MPU_ADDR = 0x68
MODEL_PATH = "modelo_rf.sav"   # tu .sav (RF o DT)

# Misma cadencia que en la ESP (~20 Hz)
SAMPLE_RATE_HZ = 20
SLEEP = 1.0 / SAMPLE_RATE_HZ

# Offsets crudos (LSB) — los mismos que usaste en ESP32-C3
AX_OFF = -4645
AY_OFF = -5769
AZ_OFF = 10448
GX_OFF = 47
GY_OFF = -148
GZ_OFF = -10

# Escalas (±2 g y ±250 dps como en tu código)
ACCEL_SENS_2G = 16384.0  # LSB por g
GYRO_SENS_250 = 131.0    # LSB por dps
G_TO_MS2 = 9.80665

# Registros mínimos
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B

def wake_up(bus):
    # Despertar el MPU6050 (sin tocar filtros ni rangos)
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)

def s16(x):
    # Convierte unsigned 16 a signed 16
    return x - 65536 if x >= 32768 else x

def read_raw_block(bus):
    # Lee 14 bytes: Ax,Ay,Az,Temp,Gx,Gy,Gz (cada uno 16 bits)
    return bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)

def read_once(bus):
    d  = read_raw_block(bus)
    ax = s16((d[0] << 8)  | d[1])  - AX_OFF
    ay = s16((d[2] << 8)  | d[3])  - AY_OFF
    az = s16((d[4] << 8)  | d[5])  - AZ_OFF
    gx = s16((d[8] << 8)  | d[9])  - GX_OFF
    gy = s16((d[10] << 8) | d[11]) - GY_OFF
    gz = s16((d[12] << 8) | d[13]) - GZ_OFF

    # Unidades físicas idénticas a tu ESP
    ax_ms2 = (ax / ACCEL_SENS_2G) * G_TO_MS2
    ay_ms2 = (ay / ACCEL_SENS_2G) * G_TO_MS2
    az_ms2 = (az / ACCEL_SENS_2G) * G_TO_MS2
    gx_dps =  gx / GYRO_SENS_250
    gy_dps =  gy / GYRO_SENS_250
    gz_dps =  gz / GYRO_SENS_250

    return ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps

def main():
    # Carga modelo (ideal: Pipeline con el mismo preprocesamiento de entrenamiento)
    clf = joblib.load(MODEL_PATH)
    has_proba = hasattr(clf, "predict_proba")

    with SMBus(I2C_BUS) as bus:
        wake_up(bus)
        print("[INFO] Leyendo y prediciendo… Ctrl+C para salir\n")
        print("time       pred     p   |    ax(ms^2)    ay(ms^2)    az(ms^2)   |   gx(dps)    gy(dps)    gz(dps)")
        print("-"*96)

        while True:
            try:
                ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps = read_once(bus)

                # Vector crudo en el ORDEN exacto del entrenamiento
                X = np.array([[ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]], dtype=float)

                y = clf.predict(X)[0]
                if has_proba:
                    proba_vec = clf.predict_proba(X)[0]
                    cls_idx = list(clf.classes_).index(y)
                    p = f"{proba_vec[cls_idx]:.2f}"
                else:
                    p = "--"

                t = time.strftime("%H:%M:%S")
                print(f"{t}  {str(y):<7} {p:>5} | "
                      f"{ax_ms2:>11.3f}  {ay_ms2:>11.3f}  {az_ms2:>11.3f} | "
                      f"{gx_dps:>9.2f}  {gy_dps:>9.2f}  {gz_dps:>9.2f}")

                time.sleep(SLEEP)
            except KeyboardInterrupt:
                print("\n[INFO] Saliendo limpio.")
                break

if __name__ == "__main__":
    main()
