#include <Arduino.h>
#include <Arduino_LSM6DSOX.h>

#define Temperature 0
#define Acceleration 1

#if Temperature
    void setup() {
        Serial.begin(115200);
        while (!Serial);  // Wait for Serial Monitor to open

        if (!IMU.begin()) {
            Serial.println("Failed to initialize IMU!");
            while (1);
        }
        Serial.println("Timestamp,Temperature(C)");
    }

    void loop() {
        float temperature;
        if (IMU.temperatureAvailable()) {
            temperature = 0;
            IMU.readTemperatureFloat(temperature);
            Serial.print(millis());      // Timestamp (ms)
            Serial.print(",");
            Serial.println(temperature); // Temperature in Celsius
        }
        delay(1000); // Log every second
    }
#endif

#if Acceleration

void setup() {
    Serial.begin(115200);
    while (!Serial);  // Wait for Serial Monitor to open

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
    Serial.println("Timestamp,Acceleration X,Acceleration Y,Acceleration Z");
}

void loop(){
    float x, y, z;
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);
        Serial.print(millis());      // Timestamp (ms)
        Serial.print(",");
        Serial.print(x); // Acceleration in X
        Serial.print(",");
        Serial.print(y); // Acceleration in Y
        Serial.print(",");
        Serial.println(z); // Acceleration in Z
    }
    delay(1000); // Log every second
}

#endif