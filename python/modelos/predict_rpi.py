#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, math, json, warnings
import numpy as np
from smbus2 import SMBus
from joblib import load

# ---------- SILENCIAR WARNINGS ----------
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning
)

# ---------- CONFIG ----------
I2C_BUS    = 1
MPU_ADDR   = 0x68
MODEL_PATH = "modelo_rf.sav"
OFFSETS_PATH = "mpu6050_offsets.json"

# Registros MPU6050
WHO_AM_I     = 0x75
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B

# Conversión (±2 g, ±250 dps)
G_TO_MS2       = 9.80665
ACC_LSB_PER_G  = 16384.0
GYR_LSB_PER_D  = 131.0

def s16(x): 
    return x - 65536 if x >= 32768 else x

def read_block(bus):
    d = bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)
    ax = s16((d[0] << 8)  | d[1])
    ay = s16((d[2] << 8)  | d[3])
    az = s16((d[4] << 8)  | d[5])
    gx = s16((d[8] << 8)  | d[9])
    gy = s16((d[10] << 8) | d[11])
    gz = s16((d[12] << 8) | d[13])
    return ax, ay, az, gx, gy, gz

def force_config(bus):
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.05)
    bus.write_byte_data(MPU_ADDR, CONFIG,       0x03)  # DLPF=3
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG,  0x00)  # ±250 dps
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)  # ±2 g
    bus.write_byte_data(MPU_ADDR, SMPLRT_DIV,   49)    # ~20 Hz
    time.sleep(0.05)

def main():
    # Cargar offsets
    with open(OFFSETS_PATH, "r") as f:
        off = json.load(f)
    AX_OFF, AY_OFF, AZ_OFF = int(off["AX_OFF"]), int(off["AY_OFF"]), int(off["AZ_OFF"])
    GX_OFF, GY_OFF, GZ_OFF = int(off["GX_OFF"]), int(off["GY_OFF"]), int(off["GZ_OFF"])

    # Cargar modelo
    model = load(MODEL_PATH)
    has_proba = hasattr(model, "predict_proba")

    with SMBus(I2C_BUS) as bus:
        who = bus.read_byte_data(MPU_ADDR, WHO_AM_I)
        if who not in (0x68, 0x69, 0x70):
            print(f"[AVISO] WHO_AM_I=0x{who:02X}, revisa conexión I2C.")
        force_config(bus)

        print("=============================================")
        print(" Predicción en tiempo real del modelo RF")
        print(" (aceleraciones en m/s², giros en °/s)")
        print("=============================================")
        print("Etiqueta | Prob | ax    ay    az    | gx    gy    gz")
        print("------------------------------------------------------")

        try:
            while True:
                ax_r, ay_r, az_r, gx_r, gy_r, gz_r = read_block(bus)

                # Aplicar offsets (LSB)
                ax = ax_r - AX_OFF
                ay = ay_r - AY_OFF
                az = az_r - AZ_OFF
                gx = gx_r - GX_OFF
                gy = gy_r - GY_OFF
                gz = gz_r - GZ_OFF

                # Convertir a unidades físicas
                ax_ms2 = (ax / ACC_LSB_PER_G) * G_TO_MS2
                ay_ms2 = (ay / ACC_LSB_PER_G) * G_TO_MS2
                az_ms2 = (az / ACC_LSB_PER_G) * G_TO_MS2
                gx_dps = gx / GYR_LSB_PER_D
                gy_dps = gy / GYR_LSB_PER_D
                gz_dps = gz / GYR_LSB_PER_D

                # Vector [ax, ay, az, gx, gy, gz]
                X = np.array([[ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]], dtype=float)

                y = model.predict(X)[0]
                p = model.predict_proba(X)[0][list(model.classes_).index(y)] if has_proba else 1.0

                print(f"{y:<9} | {p:>4.2f} | "
                      f"{ax_ms2:>5.2f} {ay_ms2:>5.2f} {az_ms2:>5.2f} | "
                      f"{gx_dps:>5.2f} {gy_dps:>5.2f} {gz_dps:>5.2f}")

                time.sleep(0.05)  # ~20 Hz
        except KeyboardInterrupt:
            print("\n[FIN] Detenido por el usuario.")

if __name__ == "__main__":
    main()
