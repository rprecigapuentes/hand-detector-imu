#include <Arduino.h>
#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <math.h>

// Pines I2C para tu ESP32-C3 SuperMini
// Tú dijiste: SDA = 8, SCL = 9
#define SDA_PIN 8
#define SCL_PIN 9

MPU6050 mpu;

// Offsets calibrados (los que mediste quieto)
int16_t ax_off = -4645;
int16_t ay_off = -5769;
int16_t az_off = 10448;
int16_t gx_off = 47;
int16_t gy_off = -148;
int16_t gz_off = -10;

// Variables crudas
int16_t ax_raw, ay_raw, az_raw;
int16_t gx_raw, gy_raw, gz_raw;

// Estado de medición
bool midiendo = false;

void setup() {
    // Serial por USB nativo del ESP32-C3
    Serial.begin(115200);
    // Inicializar I2C en pines específicos de la C3
    Wire.begin(SDA_PIN, SCL_PIN);

    // Inicializar el MPU6050
    mpu.initialize();

    // Aplicar offsets calibrados al sensor
    mpu.setXAccelOffset(ax_off);
    mpu.setYAccelOffset(ay_off);
    mpu.setZAccelOffset(az_off);
    mpu.setXGyroOffset(gx_off);
    mpu.setYGyroOffset(gy_off);
    mpu.setZGyroOffset(gz_off);

    // Mensaje inicial (líneas que empiezan con # tu script las ignora)
    Serial.println("# Listo. Presiona 's' para iniciar y 'e' para detener medición.");
    Serial.println("# Formato de salida: t_ms,ax_ms2,ay_ms2,az_ms2,a_total_ms2,gx_dps,gy_dps,gz_dps");
}

void loop() {

    // Leer comandos desde el USB serial (Python o tú manualmente)
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
        // Leer datos crudos del acelerómetro y giroscopio
        mpu.getAcceleration(&ax_raw, &ay_raw, &az_raw);
        mpu.getRotation(&gx_raw, &gy_raw, &gz_raw);

        // Convertir a unidades físicas
        const float g_ms2 = 9.80665f;

        // Aceleración en m/s^2
        float ax_ms2 = ((float)ax_raw / 16384.0f) * g_ms2;
        float ay_ms2 = ((float)ay_raw / 16384.0f) * g_ms2;
        float az_ms2 = ((float)az_raw / 16384.0f) * g_ms2;

        // Magnitud total de aceleración
        float a_total_ms2 = sqrtf(ax_ms2 * ax_ms2 +
                                  ay_ms2 * ay_ms2 +
                                  az_ms2 * az_ms2);

        // Velocidad angular en grados/seg
        float gx_dps = ((float)gx_raw) / 131.0f;
        float gy_dps = ((float)gy_raw) / 131.0f;
        float gz_dps = ((float)gz_raw) / 131.0f;

        // Timestamp
        unsigned long t_ms = millis();

        // Imprimir una línea CSV que Python entiende directamente
        Serial.print(t_ms);       Serial.print(",");
        Serial.print(ax_ms2, 3);  Serial.print(",");
        Serial.print(ay_ms2, 3);  Serial.print(",");
        Serial.print(az_ms2, 3);  Serial.print(",");
        Serial.print(a_total_ms2, 3);  Serial.print(",");
        Serial.print(gx_dps, 3);  Serial.print(",");
        Serial.print(gy_dps, 3);  Serial.print(",");
        Serial.println(gz_dps, 3);

        // Frecuencia de muestreo ~20 Hz (cada 50 ms)
        delay(50);

    } else {
        // Para no quemar CPU cuando está "parado"
        delay(100);
    }
}
