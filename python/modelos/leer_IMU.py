# Archivo: imu_read_calibrated.py
# Raspberry Pi + MPU6050 por I2C. Fuerza ±2 g y ±250 dps, aplica los mismos offsets
# que usas en Arduino y emite CSV: t_ms,ax,ay,az,a_total,gx,gy,gz

from smbus2 import SMBus
from time import sleep, time
from math import sqrt

# Dirección I2C (AD0 a GND -> 0x68)
MPU_ADDR = 0x68

# Registros clave
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
WHO_AM_I     = 0x75

# Offsets calibrados (los mismos del sketch de Arduino)
ax_off, ay_off, az_off = -4645, -5769, 10448
gx_off, gy_off, gz_off = 47, -148, -10

# Factores para ±2 g y ±250 dps
G_MS2       = 9.80665
ACC_LSB_PER_G  = 16384.0   # ±2 g
GYRO_LSB_PER_D = 131.0     # ±250 dps

def read_word(bus, reg):
    hi = bus.read_byte_data(MPU_ADDR, reg)
    lo = bus.read_byte_data(MPU_ADDR, reg + 1)
    val = (hi << 8) | lo
    if val & 0x8000:
        val = -((65535 - val) + 1)
    return val

def init_mpu(bus):
    # Despertar
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    sleep(0.05)
    # Filtro DLPF moderado (CONFIG), tasa muestreo ~1kHz/(1+div)
    bus.write_byte_data(MPU_ADDR, CONFIG, 0x03)        # DLPF = 3 (~44 Hz accel, ~42 Hz gyro)
    bus.write_byte_data(MPU_ADDR, SMPLRT_DIV, 0x04)    # 1kHz/(1+4)=200 Hz interno
    # Fuerza rangos: gyro ±250 dps, accel ±2 g
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG,  0x00)  # FS_SEL=0 -> ±250 dps
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)  # AFS_SEL=0 -> ±2 g
    sleep(0.05)

def read_calibrated(bus):
    # Crudos
    ax = read_word(bus, ACCEL_XOUT_H)
    ay = read_word(bus, ACCEL_XOUT_H + 2)
    az = read_word(bus, ACCEL_XOUT_H + 4)
    gx = read_word(bus, GYRO_XOUT_H)
    gy = read_word(bus, GYRO_XOUT_H + 2)
    gz = read_word(bus, GYRO_XOUT_H + 4)

    # Aplica los mismos offsets que en Arduino (sustracción en cuentas)
    ax -= ax_off
    ay -= ay_off
    az -= az_off
    gx -= gx_off
    gy -= gy_off
    gz -= gz_off

    # A m/s² y dps
    ax_ms2 = (ax / ACC_LSB_PER_G) * G_MS2
    ay_ms2 = (ay / ACC_LSB_PER_G) * G_MS2
    az_ms2 = (az / ACC_LSB_PER_G) * G_MS2
    a_tot  = sqrt(ax_ms2*ax_ms2 + ay_ms2*ay_ms2 + az_ms2*az_ms2)

    gx_dps = gx / GYRO_LSB_PER_D
    gy_dps = gy / GYRO_LSB_PER_D
    gz_dps = gz / GYRO_LSB_PER_D

    return ax_ms2, ay_ms2, az_ms2, a_tot, gx_dps, gy_dps, gz_dps

def main():
    with SMBus(1) as bus:
        # Comprobación rápida
        who = bus.read_byte_data(MPU_ADDR, WHO_AM_I)
        if who not in (0x68, 0x69):
            print(f"# WHO_AM_I inesperado: 0x{who:02X}. Revisa la dirección o el cableado.")
        init_mpu(bus)

        print("# Leyendo MPU6050 (±2 g, ±250 dps) con offsets estilo Arduino")
        print("# Formato: t_ms,ax_ms2,ay_ms2,az_ms2,a_total_ms2,gx_dps,gy_dps,gz_dps")

        t0 = time()
        try:
            while True:
                t_ms = int((time() - t0) * 1000)
                ax, ay, az, a_tot, gx, gy, gz = read_calibrated(bus)
                print(f"{t_ms},{ax:.3f},{ay:.3f},{az:.3f},{a_tot:.3f},{gx:.3f},{gy:.3f},{gz:.3f}")
                sleep(0.05)  # ~20 Hz
        except KeyboardInterrupt:
            print("\n# Fin de medición")

if __name__ == "__main__":
    main()
