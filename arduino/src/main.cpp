#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <math.h>

MPU6050 mpu;

// Offsets calibrados (ajústalos si usas otro módulo)
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
  Serial.begin(115200);
  Wire.begin();  // En Arduino los pines I2C son fijos: A4 (SDA), A5 (SCL)

  // Inicializar el MPU6050
  mpu.initialize();

  // Aplicar offsets calibrados
  mpu.setXAccelOffset(ax_off);
  mpu.setYAccelOffset(ay_off);
  mpu.setZAccelOffset(az_off);
  mpu.setXGyroOffset(gx_off);
  mpu.setYGyroOffset(gy_off);
  mpu.setZGyroOffset(gz_off);

  Serial.println("# Listo. Presiona 's' para iniciar y 'e' para detener medición.");
  Serial.println("# Formato: t_ms,ax_ms2,ay_ms2,az_ms2,a_total_ms2,gx_dps,gy_dps,gz_dps");
}

void loop() {
  // Leer comandos desde el monitor serial
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
    // Leer datos crudos del sensor
    mpu.getAcceleration(&ax_raw, &ay_raw, &az_raw);
    mpu.getRotation(&gx_raw, &gy_raw, &gz_raw);

    const float g_ms2 = 9.80665f;

    // Aceleración en m/s² (rango ±2g => 16384 LSB/g)
    float ax_ms2 = (ax_raw / 16384.0f) * g_ms2;
    float ay_ms2 = (ay_raw / 16384.0f) * g_ms2;
    float az_ms2 = (az_raw / 16384.0f) * g_ms2;

    // Magnitud total de la aceleración
    float a_total_ms2 = sqrtf(ax_ms2 * ax_ms2 +
                              ay_ms2 * ay_ms2 +
                              az_ms2 * az_ms2);

    // Velocidad angular en grados/seg (rango ±250°/s => 131 LSB/dps)
    float gx_dps = gx_raw / 131.0f;
    float gy_dps = gy_raw / 131.0f;
    float gz_dps = gz_raw / 131.0f;

    unsigned long t_ms = millis();

    // Imprimir una línea CSV
    Serial.print(t_ms);            Serial.print(",");
    Serial.print(ax_ms2, 3);       Serial.print(",");
    Serial.print(ay_ms2, 3);       Serial.print(",");
    Serial.print(az_ms2, 3);       Serial.print(",");
    Serial.print(a_total_ms2, 3);  Serial.print(",");
    Serial.print(gx_dps, 3);       Serial.print(",");
    Serial.print(gy_dps, 3);       Serial.print(",");
    Serial.println(gz_dps, 3);

    delay(50);  // Aproximadamente 20 Hz
  } else {
    delay(100);
  }
}
