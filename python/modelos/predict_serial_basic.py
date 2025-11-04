#!/usr/bin/env python3
import time
import warnings
import joblib
import numpy as np
from smbus2 import SMBus

# Silenciar el warning de sklearn: "X does not have valid feature names..."
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning
)

# =============================
# CONFIG
# =============================
I2C_BUS = 1
MPU_ADDR = 0x68

MODEL_PATH = "modelo_rf.sav"   # cambia por tu .sav (RF o DT)
SAMPLE_RATE_HZ = 20            # ~20 Hz como en tu ESP (delay 50 ms)
SLEEP = 1.0 / SAMPLE_RATE_HZ

# Offsets crudos (LSB), iguales a tu ESP32-C3
AX_OFF = -4645
AY_OFF = -5769
AZ_OFF = 10448
GX_OFF = 47
GY_OFF = -148
GZ_OFF = -10

# Escalas ±2 g y ±250 dps
ACCEL_SENS_2G = 16384.0
GYRO_SENS_250 = 131.0
G_TO_MS2 = 9.80665

# Registros MPU6050
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B

def mpu6050_init(bus):
    # Despertar
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    # Filtro DLPF ~94 Hz
    bus.write_byte_data(MPU_ADDR, CONFIG, 0x02)
    # Gyro ±250 dps
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG, 0x00)
    # Accel ±2 g
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)
    # Sample rate = 1kHz/(1+div) con DLPF ON
    div = max(0, int(1000 / SAMPLE_RATE_HZ) - 1)
    bus.write_byte_data(MPU_ADDR, SMPLRT_DIV, div)
    time.sleep(0.05)

def s16(x):
    return x - 65536 if x >= 32768 else x

def read_mpu6050(bus):
    d = bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)
    ax = s16((d[0] << 8)  | d[1])  - AX_OFF
    ay = s16((d[2] << 8)  | d[3])  - AY_OFF
    az = s16((d[4] << 8)  | d[5])  - AZ_OFF
    gx = s16((d[8] << 8)  | d[9])  - GX_OFF
    gy = s16((d[10] << 8) | d[11]) - GY_OFF
    gz = s16((d[12] << 8) | d[13]) - GZ_OFF

    ax_ms2 = (ax / ACCEL_SENS_2G) * G_TO_MS2
    ay_ms2 = (ay / ACCEL_SENS_2G) * G_TO_MS2
    az_ms2 = (az / ACCEL_SENS_2G) * G_TO_MS2

    gx_dps = gx / GYRO_SENS_250
    gy_dps = gy / GYRO_SENS_250
    gz_dps = gz / GYRO_SENS_250

    return ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps

def main():
    # Cargar modelo
    clf = joblib.load(MODEL_PATH)
    has_proba = hasattr(clf, "predict_proba")

    with SMBus(I2C_BUS) as bus:
        mpu6050_init(bus)
        print("[INFO] Leyendo y prediciendo… Ctrl+C para salir\n")
        print("time       pred     p   |    ax(ms^2)    ay(ms^2)    az(ms^2)   |   gx(dps)    gy(dps)    gz(dps)")
        print("-"*96)

        while True:
            try:
                ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps = read_mpu6050(bus)

                # ndarray puro, sin pandas
                X = np.array([[ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]], dtype=float)

                y = clf.predict(X)[0]
                if has_proba:
                    proba_vec = clf.predict_proba(X)[0]
                    # prob de la clase predicha
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
