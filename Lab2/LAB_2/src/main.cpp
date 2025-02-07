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
    #include <TimeLib.h> // Include the Time library
    // Define the starting date and time
    time_t startTime;
    unsigned long startMillis;

    void setup() {
        Serial.begin(115200);
        while (!Serial);  // Wait for Serial Monitor to open

        if (!IMU.begin()) {
            Serial.println("Failed to initialize IMU!");
            while (1);
        }
        Serial.println("Date,Time,Ax,Ay,Az,Gx,Gy,Gz");
        startMillis = millis();
        startTime = now(); // Capture the start time
    }

    void loop() {
        float ax, ay, az;
        float gx, gy, gz;
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
            IMU.readAcceleration(ax, ay, az);
            IMU.readGyroscope(gx, gy, gz);

            unsigned long currentMillis = millis();
            time_t currentTime = startTime + (currentMillis - startMillis) / 1000;

            char timeBuffer[9];
            snprintf(timeBuffer, sizeof(timeBuffer), "%02d:%02d:%02d", hour(currentTime), minute(currentTime), second(currentTime));

            // Print the date and time
            Serial.print("2025-02-07");
            Serial.print(",");
            Serial.print(timeBuffer);
            Serial.print(",");
            Serial.print(ax); // Acceleration in X
            Serial.print(",");
            Serial.print(ay); // Acceleration in Y
            Serial.print(",");
            Serial.print(az); // Acceleration in Z
            Serial.print(",");
            Serial.print(gx); // Gyroscope in X
            Serial.print(",");
            Serial.print(gy); // Gyroscope in Y
            Serial.print(",");
            Serial.println(gz); // Gyroscope in Z
        }
        delay(500); // Log every 0.5 second
    }
#endif