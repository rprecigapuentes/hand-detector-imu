import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from collections import deque
import csv
import time
import os
import sys

# ================================
# CONFIGURACIÓN
# ================================
BAUD_RATE = 115200
MAX_POINTS = 200  # puntos visibles en la gráfica
CSV_PATH = "data/data.csv"

# ================================
# DETECTAR PUERTO SERIAL
# ================================
def detectar_puerto():
    puertos = list(serial.tools.list_ports.comports())
    if not puertos:
        raise RuntimeError("No se detectó ningún puerto serial. Conecta el Arduino.")
    for p in puertos:
        if "ACM" in p.device or "USB" in p.device:
            print(f"[✔] Detectado posible Arduino en {p.device}")
            return p.device
    print(f"[!] No vi ACM/USB explícito, usando {puertos[0].device}")
    return puertos[0].device

# ================================
# CONFIGURAR CSV
# ================================
def preparar_csv(ruta_csv):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    f = open(ruta_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_ms","ax_ms2","ay_ms2","az_ms2","a_total_ms2","gx_dps","gy_dps","gz_dps"])
    return f, w

# ================================
# CONFIGURAR GRÁFICA EN TIEMPO REAL
# ================================
def preparar_grafica(max_points):
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    data_ax, data_ay, data_az = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)
    data_gx, data_gy, data_gz = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)

    # Subplot 1: Aceleraciones
    line_ax, = axes[0].plot([], [], label="Ax [m/s²]")
    line_ay, = axes[0].plot([], [], label="Ay [m/s²]")
    line_az, = axes[0].plot([], [], label="Az [m/s²]")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Aceleración (m/s²)")
    axes[0].set_ylim(-20, 20)
    axes[0].set_title("MPU6050 en tiempo real")

    # Subplot 2: Giroscopio
    line_gx, = axes[1].plot([], [], label="Gx [°/s]")
    line_gy, = axes[1].plot([], [], label="Gy [°/s]")
    line_gz, = axes[1].plot([], [], label="Gz [°/s]")
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Muestras recientes")
    axes[1].set_ylabel("Velocidad angular (°/s)")
    axes[1].set_ylim(-250, 250)

    plt.tight_layout()

    return {
        "fig": fig,
        "axes": axes,
        "data_ax": data_ax, "data_ay": data_ay, "data_az": data_az,
        "data_gx": data_gx, "data_gy": data_gy, "data_gz": data_gz,
        "line_ax": line_ax, "line_ay": line_ay, "line_az": line_az,
        "line_gx": line_gx, "line_gy": line_gy, "line_gz": line_gz
    }

def actualizar_grafica(g):
    g["line_ax"].set_ydata(g["data_ax"])
    g["line_ay"].set_ydata(g["data_ay"])
    g["line_az"].set_ydata(g["data_az"])
    g["line_gx"].set_ydata(g["data_gx"])
    g["line_gy"].set_ydata(g["data_gy"])
    g["line_gz"].set_ydata(g["data_gz"])

    x_acc = range(len(g["data_ax"]))
    x_gyro = range(len(g["data_gx"]))
    for l in ("line_ax", "line_ay", "line_az"):
        g[l].set_xdata(x_acc)
    for l in ("line_gx", "line_gy", "line_gz"):
        g[l].set_xdata(x_gyro)

    for ax in g["axes"]:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)

    plt.pause(0.001)

# ================================
# LOOP DE ADQUISICIÓN
# ================================
def correr_adquisicion(ser, writer, graf):
    print("\n[INFO] Presiona ENTER para INICIAR medición...")
    input()

    ser.write(b's')
    print("[OK] Envié 's'. Leyendo datos.\n")
    print("Ctrl+C para detener cuando quieras.\n")

    try:
        while True:
            linea = ser.readline().decode("utf-8", errors="ignore").strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split(",")
            if len(partes) != 8:
                continue

            try:
                t_ms, ax_ms2, ay_ms2, az_ms2, a_total_ms2, gx_dps, gy_dps, gz_dps = map(float, partes)
            except ValueError:
                continue

            writer.writerow([t_ms, ax_ms2, ay_ms2, az_ms2, a_total_ms2, gx_dps, gy_dps, gz_dps])

            graf["data_ax"].append(ax_ms2)
            graf["data_ay"].append(ay_ms2)
            graf["data_az"].append(az_ms2)
            graf["data_gx"].append(gx_dps)
            graf["data_gy"].append(gy_dps)
            graf["data_gz"].append(gz_dps)

            actualizar_grafica(graf)

    except KeyboardInterrupt:
        print("\n[STOP] Captura detenida por el usuario con Ctrl+C.\n")
        try:
            ser.write(b'e')
            time.sleep(0.2)
        except Exception:
            pass

# ================================
# MAIN
# ================================
def main():
    puerto = detectar_puerto()
    print(f"[INFO] Abriendo puerto {puerto} a {BAUD_RATE} baud...")
    ser = serial.Serial(puerto, BAUD_RATE, timeout=1)
    time.sleep(2)

    csv_file, writer = preparar_csv(CSV_PATH)
    graf = preparar_grafica(MAX_POINTS)

    try:
        correr_adquisicion(ser, writer, graf)

    except KeyboardInterrupt:
        print("\n[STOP] Interrupción manual recibida.")
        try:
            ser.write(b'e')
        except Exception:
            pass

    finally:
        try:
            ser.close()
        except Exception:
            pass
        try:
            csv_file.close()
        except Exception:
            pass
        plt.close("all")   # <- cierra la ventana de Matplotlib sin errores
        print(f"[✔] Puerto cerrado.")
        print(f"[✔] CSV guardado en {os.path.abspath(CSV_PATH)}")
        print("[FIN] Ejecución terminada correctamente.")

if __name__ == "__main__":
    main()
