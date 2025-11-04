#!/usr/bin/env python3
import time
import joblib
import numpy as np
from smbus2 import SMBus

# =============================
# CONFIG
# =============================
I2C_BUS = 1          # GPIO2 (SDA) y GPIO3 (SCL) usan bus 1
MPU_ADDR = 0x68

MODEL_PATH = "modelo_rf.sav"   # cambia por tu .sav (RF o DT)
SAMPLE_RATE_HZ = 20            # ~20 Hz como en tu ESP (delay 50 ms)
SLEEP = 1.0 / SAMPLE_RATE_HZ

# Offsets crudos medidos (los mismos de la ESP, en LSB)
AX_OFF = -4645
AY_OFF = -5769
AZ_OFF = 10448
GX_OFF = 47
GY_OFF = -148
GZ_OFF = -10

# Escalas elegidas: ±2 g y ±250 dps
ACCEL_SENS_2G = 16384.0  # LSB/g
GYRO_SENS_250 = 131.0    # LSB/dps
G_TO_MS2 = 9.80665

# Orden EXACTO de features del entrenamiento:
# ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps
# =============================
# REGISTROS MPU6050
# =============================
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
    # Filtro DLPF: BW ~94 Hz
    bus.write_byte_data(MPU_ADDR, CONFIG, 0x02)
    # Gyro ±250 dps
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG, 0x00)
    # Accel ±2 g
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)
    # Sample rate = 1 kHz / (1 + div) con DLPF ON
    div = max(0, int(1000 / SAMPLE_RATE_HZ) - 1)
    bus.write_byte_data(MPU_ADDR, SMPLRT_DIV, div)
    time.sleep(0.05)

def s16(x):
    return x - 65536 if x >= 32768 else x

def read_mpu6050(bus):
    # Lee 14 bytes (Ax,Ay,Az,T,Gx,Gy,Gz)
    data = bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)
    ax = s16((data[0] << 8) | data[1])
    ay = s16((data[2] << 8) | data[3])
    az = s16((data[4] << 8) | data[5])
    gx = s16((data[8] << 8) | data[9])
    gy = s16((data[10] << 8) | data[11])
    gz = s16((data[12] << 8) | data[13])

    # Aplica offsets en crudo (mismos que en la ESP)
    ax -= AX_OFF
    ay -= AY_OFF
    az -= AZ_OFF
    gx -= GX_OFF
    gy -= GY_OFF
    gz -= GZ_OFF

    # Convierte a unidades físicas
    ax_ms2 = (ax / ACCEL_SENS_2G) * G_TO_MS2
    ay_ms2 = (ay / ACCEL_SENS_2G) * G_TO_MS2
    az_ms2 = (az / ACCEL_SENS_2G) * G_TO_MS2

    gx_dps = gx / GYRO_SENS_250
    gy_dps = gy / GYRO_SENS_250
    gz_dps = gz / GYRO_SENS_250

    return ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps

def main():
    # Carga del modelo (ideal: Pipeline con escalado)
    print(f"[INFO] Cargando modelo: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    if not hasattr(clf, "predict"):
        raise TypeError("El objeto cargado no tiene .predict(). ¿Guardaste el Pipeline/clasificador correcto?")
    print("[INFO] Modelo cargado.")

    with SMBus(I2C_BUS) as bus:
        print("[INFO] Inicializando MPU6050...")
        mpu6050_init(bus)
        print("[INFO] Listo. Leyendo y prediciendo... [Ctrl+C para salir]")

        try:
            while True:
                ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps = read_mpu6050(bus)

                # Vector en el mismo orden del entrenamiento
                X = np.array([[ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]], dtype=float)

                y = clf.predict(X)[0]
                print(f"Pred: {y} | "
                      f"ax={ax_ms2:.3f} ay={ay_ms2:.3f} az={az_ms2:.3f} ms^2 | "
                      f"gx={gx_dps:.2f} gy={gy_dps:.2f} gz={gz_dps:.2f} dps")

                time.sleep(SLEEP)
        except KeyboardInterrupt:
            print("\n[INFO] Saliendo limpio.")

if __name__ == "__main__":
    main()
