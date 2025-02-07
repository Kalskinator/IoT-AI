import serial
import time
import csv
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os

#Suppress Tk warning
os.environ["TK_SILENCE_DEPRECATION"] = "1"

#Initialize serial connection
ser = serial.Serial('/dev/cu.usbmodem101', baudrate=115200, timeout=1)
ser.flushInput()

# # Plot parameters
# plot_window = 20
# y_var = np.zeros(plot_window)  # Preallocate fixed-size array
# # # Set up Matplotlib figure
# plt.ion()  # Enable interactive mode
# fig, ax = plt.subplots()
# line, = ax.plot(y_var)
# ax.set_ylim(0, 1023)  # Adjust y-limits if necessary (e.g., for sensor data)
while True:
    try:
        # Read serial data
        ser_bytes = ser.readline()
        raw_data = ser_bytes.decode("utf-8").strip()
        print(f"Raw data: {raw_data}")

        # Parse data as a list of floats
        try:
            values = [float(value) for value in raw_data.split(',')]
            print(f"Parsed values: {values}")
        except ValueError:
            print("Error: Could not parse some values as float. Skipping.")
            continue

        # Save data to CSV
        timestamp = time.time()
        with open("lab2/Data/temperature.csv", "a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow( values)  # Combine timestamp with values into one row

    except KeyboardInterrupt:
        print("Exiting program due to Keyboard Interrupt.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break