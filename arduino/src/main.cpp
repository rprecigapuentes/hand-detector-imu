#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
#include <math.h>

MPU6050 mpu;

// Offsets calibrados
int16_t ax_off = -4645;
int16_t ay_off = -5769;
int16_t az_off = 10448;
int16_t gx_off = 47;
int16_t gy_off = -148;
int16_t gz_off = -10;

// Variables
int16_t ax_raw, ay_raw, az_raw;
int16_t gx_raw, gy_raw, gz_raw;
bool midiendo = false;

void setup() {
    Serial.begin(115200);
    Wire.begin();
    mpu.initialize();

    // Asignar offsets calibrados
    mpu.setXAccelOffset(ax_off);
    mpu.setYAccelOffset(ay_off);
    mpu.setZAccelOffset(az_off);
    mpu.setXGyroOffset(gx_off);
    mpu.setYGyroOffset(gy_off);
    mpu.setZGyroOffset(gz_off);

    // Pequeño aviso solo al inicio (Python ignora las primeras líneas)
    Serial.println("# Listo. Presiona 's' para iniciar y 'e' para detener medición.");
    Serial.println("# Formato de salida: t_ms,ax_ms2,ay_ms2,az_ms2,a_total_ms2,gx_dps,gy_dps,gz_dps");
}

void loop() {
    // Control por teclado
    if (Serial.available()) {
        char comando = Serial.read();
        if (comando == 's') {
            midiendo = true;
            Serial.println("# Inicio de medición");
        } else if (comando == 'e') {
            midiendo = false;
            Serial.println("# Fin de medición");
        }
    }

    if (midiendo) {
        // Leer datos crudos
        mpu.getAcceleration(&ax_raw, &ay_raw, &az_raw);
        mpu.getRotation(&gx_raw, &gy_raw, &gz_raw);

        // Conversión a unidades físicas
        const float g_ms2 = 9.80665;
        float ax_ms2 = ((float)ax_raw / 16384.0) * g_ms2;
        float ay_ms2 = ((float)ay_raw / 16384.0) * g_ms2;
        float az_ms2 = ((float)az_raw / 16384.0) * g_ms2;
        float a_total_ms2 = sqrt(ax_ms2 * ax_ms2 + ay_ms2 * ay_ms2 + az_ms2 * az_ms2);

        float gx_dps = ((float)gx_raw) / 131.0;
        float gy_dps = ((float)gy_raw) / 131.0;
        float gz_dps = ((float)gz_raw) / 131.0;

        unsigned long t_ms = millis();

        // Imprimir línea CSV pura
        Serial.print(t_ms); Serial.print(",");
        Serial.print(ax_ms2, 3); Serial.print(",");
        Serial.print(ay_ms2, 3); Serial.print(",");
        Serial.print(az_ms2, 3); Serial.print(",");
        Serial.print(a_total_ms2, 3); Serial.print(",");
        Serial.print(gx_dps, 3); Serial.print(",");
        Serial.print(gy_dps, 3); Serial.print(",");
        Serial.println(gz_dps, 3);

        delay(50); // 20 Hz
    } else {
        delay(100);
    }
}
