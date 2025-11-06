# filter_gestures.py
import csv
from pathlib import Path

INPUT  = "dataset_labeled.csv"              # pon aquí tu archivo de origen
OUTPUT = "dataset_nada_like.csv"     # y aquí el nombre del archivo de salida

KEEP = {"nada", "like"}  # clases a conservar (minúsculas)

def main():
    in_path = Path(INPUT)
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {INPUT}. Pon el CSV en esta carpeta o cambia la ruta.")

    # utf-8-sig para tolerar BOM si el archivo viene de Excel
    with open(INPUT, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if "gesture" not in (reader.fieldnames or []):
            raise ValueError("El CSV no tiene la columna 'gesture' en el encabezado.")

        with open(OUTPUT, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            kept = 0
            for row in reader:
                gesture = (row.get("gesture") or "").strip().lower()
                if gesture in KEEP:
                    writer.writerow(row)
                    kept += 1

    print(f"Hecho. Guardado {OUTPUT} con {kept} filas filtradas de 'nada' y 'like'.")

if __name__ == "__main__":
    main()
