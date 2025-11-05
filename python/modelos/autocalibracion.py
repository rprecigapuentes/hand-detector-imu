#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autocalibración y lectura calibrada de MPU6050 en Raspberry Pi (I2C).
- Fuerza ±2 g y ±250 dps, DLPF=3, sample ~20 Hz.
- Detecta automáticamente el eje de gravedad y calcula offsets (LSB).
- Guarda/lee offsets en 'mpu6050_offsets.json'.
- Emite CSV: t_ms,ax,ay,az,a_total,gx,gy,gz

Requisitos:
  sudo apt install python3-smbus
  pip install smbus2

Conexión:
  VCC 3.3V, GND, SDA=GPIO2(pin3), SCL=GPIO3(pin5), AD0=GND (dir 0x68)
"""

import json
import time
import math
from smbus2 import SMBus

I2C_BUS   = 1
MPU_ADDR  = 0x68

# Registros clave
WHO_AM_I     = 0x75
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43

# Constantes físicas y escalas por rango
G_TO_MS2 = 9.80665
ACC_LSB_PER_G = {0: 16384.0, 1: 8192.0, 2: 4096.0, 3: 2048.0}   # FS_a = 0..3
GYR_LSB_PER_D = {0: 131.0,   1: 65.5,   2: 32.8,   3: 16.4}     # FS_g = 0..3

OFFSETS_PATH = "mpu6050_offsets.json"

def s16(x): 
    return x - 65536 if x >= 32768 else x

def mpu_write(bus, reg, val):
    bus.write_byte_data(MPU_ADDR, reg, val)

def mpu_read(bus, reg):
    return bus.read_byte_data(MPU_ADDR, reg)

def read_block(bus):
    # Lee 14 bytes: Ax,Ay,Az,Temp,Gx,Gy,Gz
    d = bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)
    ax = s16((d[0] << 8)  | d[1])
    ay = s16((d[2] << 8)  | d[3])
    az = s16((d[4] << 8)  | d[5])
    gx = s16((d[8] << 8)  | d[9])
    gy = s16((d[10] << 8) | d[11])
    gz = s16((d[12] << 8) | d[13])
    return ax, ay, az, gx, gy, gz

def force_config(bus):
    # Despertar
    mpu_write(bus, PWR_MGMT_1, 0x00)
    time.sleep(0.05)
    # Filtro digital: DLPF=3 (~44 Hz accel, ~42 Hz gyro)
    mpu_write(bus, CONFIG, 0x03)
    # Rangos: gyro ±250 dps (FS_SEL=0), accel ±2 g (AFS_SEL=0)
    mpu_write(bus, GYRO_CONFIG,  0x00)
    mpu_write(bus, ACCEL_CONFIG, 0x00)
    # Sample rate ~20 Hz (con DLPF activo, base 1 kHz)
    mpu_write(bus, SMPLRT_DIV, 49)
    time.sleep(0.05)

def read_scales(bus):
    ac = mpu_read(bus, ACCEL_CONFIG)
    gc = mpu_read(bus, GYRO_CONFIG)
    fs_a = (ac >> 3) & 0x03
    fs_g = (gc >> 3) & 0x03
    return ACC_LSB_PER_G[fs_a], GYR_LSB_PER_D[fs_g], fs_a, fs_g

def stddev(vals):
    if not vals:
        return 0.0
    m = sum(vals)/len(vals)
    return (sum((v-m)*(v-m) for v in vals)/len(vals))**0.5

def autocalibrate(bus, seconds=5.0, hz=100.0):
    """
    Autocalibración en LSB. Detecta eje de gravedad automáticamente.
    Offset aplicado por sustracción: corrected = raw - offset.
    Tarjetas: 
      - eje con |g| mayor: target = ±ACC_LSB (según signo)
      - ejes restantes: target = 0
      - gyro: target = 0 en los tres ejes
    """
    print("# Coloca el IMU QUIETO. Autocalibrando...")
    samples = int(seconds * hz)
    dt = 1.0 / hz

    ax_ls, ay_ls, az_ls = [], [], []
    gx_ls, gy_ls, gz_ls = [], [], []

    for _ in range(samples):
        ax, ay, az, gx, gy, gz = read_block(bus)
        ax_ls.append(ax); ay_ls.append(ay); az_ls.append(az)
        gx_ls.append(gx); gy_ls.append(gy); gz_ls.append(gz)
        time.sleep(dt)

    # Medias
    ax_m = sum(ax_ls)/samples
    ay_m = sum(ay_ls)/samples
    az_m = sum(az_ls)/samples
    gx_m = sum(gx_ls)/samples
    gy_m = sum(gy_ls)/samples
    gz_m = sum(gz_ls)/samples

    # Desviaciones para verificar que estuviste quieto
    ax_sd = stddev(ax_ls); ay_sd = stddev(ay_ls); az_sd = stddev(az_ls)
    if max(ax_sd, ay_sd, az_sd) > 200:  # heurístico en LSB
        print(f"# WARNING: Mucho movimiento durante la calibración (std max={max(ax_sd,ay_sd,az_sd):.1f} LSB). "
              f"Repite con el sensor inmóvil para mejor resultado.")

    # Determina eje de gravedad
    acc_lsb, gyr_lsb, fs_a, fs_g = read_scales(bus)
    # Con ±2 g, target de gravedad en LSB:
    g_target = acc_lsb  # ~16384

    means = [ax_m, ay_m, az_m]
    idx = max(range(3), key=lambda i: abs(means[i]))
    signs = [1 if m >= 0 else -1 for m in means]
    targets = [0.0, 0.0, 0.0]
    targets[idx] = signs[idx] * g_target

    # Offsets en LSB tal que raw - off ≈ target
    ax_off = ax_m - targets[0]
    ay_off = ay_m - targets[1]
    az_off = az_m - targets[2]
    gx_off = gx_m  # queremos 0
    gy_off = gy_m
    gz_off = gz_m

    offsets = {
        "AX_OFF": int(round(ax_off)),
        "AY_OFF": int(round(ay_off)),
        "AZ_OFF": int(round(az_off)),
        "GX_OFF": int(round(gx_off)),
        "GY_OFF": int(round(gy_off)),
        "GZ_OFF": int(round(gz_off)),
        "FS_A":   fs_a,
        "FS_G":   fs_g
    }

    with open(OFFSETS_PATH, "w") as f:
        json.dump(offsets, f, indent=2)
    print("# Offsets guardados en", OFFSETS_PATH)
    print("# Offsets (LSB):", offsets)
    return offsets, acc_lsb, gyr_lsb

def load_offsets(acc_lsb, gyr_lsb):
    try:
        with open(OFFSETS_PATH, "r") as f:
            off = json.load(f)
        # Si los rangos cambiaron, seguimos usando LSB; no pasa nada.
        return off
    except FileNotFoundError:
        # Si no hay archivo, usamos ceros
        return {
            "AX_OFF": 0, "AY_OFF": 0, "AZ_OFF": 0,
            "GX_OFF": 0, "GY_OFF": 0, "GZ_OFF": 0
        }

def main():
    with SMBus(I2C_BUS) as bus:
        who = mpu_read(bus, WHO_AM_I)
        if who not in (0x68, 0x69, 0x70):  # algunos clones reportan 0x70
            print(f"# WHO_AM_I=0x{who:02X} extraño. Sigo, pero revisa dirección/cableado.")
        force_config(bus)
        acc_lsb, gyr_lsb, fs_a, fs_g = read_scales(bus)
        print(f"# FS accel={['±2g','±4g','±8g','±16g'][fs_a]} (LSB/g={acc_lsb}), "
              f"gyro={['±250','±500','±1000','±2000'][fs_g]} (LSB/°s={gyr_lsb})")

        # Autocalibrar siempre al arrancar. Si no quieres, comenta las 2 líneas siguientes:
        offsets, acc_lsb, gyr_lsb = autocalibrate(bus, seconds=5.0, hz=100.0)
        # offsets = load_offsets(acc_lsb, gyr_lsb)  # alternativo: cargar existentes

        print("# Lectura calibrada. CSV: t_ms,ax,ay,az,a_total,gx,gy,gz")
        t0 = time.time()
        try:
            while True:
                ax_r, ay_r, az_r, gx_r, gy_r, gz_r = read_block(bus)

                ax = ax_r - offsets["AX_OFF"]
                ay = ay_r - offsets["AY_OFF"]
                az = az_r - offsets["AZ_OFF"]
                gx = gx_r - offsets["GX_OFF"]
                gy = gy_r - offsets["GY_OFF"]
                gz = gz_r - offsets["GZ_OFF"]

                ax_ms2 = (ax / acc_lsb) * G_TO_MS2
                ay_ms2 = (ay / acc_lsb) * G_TO_MS2
                az_ms2 = (az / acc_lsb) * G_TO_MS2
                a_tot  = math.sqrt(ax_ms2*ax_ms2 + ay_ms2*ay_ms2 + az_ms2*az_ms2)

                gx_dps = gx / gyr_lsb
                gy_dps = gy / gyr_lsb
                gz_dps = gz / gyr_lsb

                t_ms = int((time.time() - t0) * 1000)
                print(f"{t_ms},{ax_ms2:.3f},{ay_ms2:.3f},{az_ms2:.3f},{a_tot:.3f},"
                      f"{gx_dps:.3f},{gy_dps:.3f},{gz_dps:.3f}")
                time.sleep(0.05)  # ~20 Hz
        except KeyboardInterrupt:
            print("\n# Fin.")

if __name__ == "__main__":
    main()
