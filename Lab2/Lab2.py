import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r"""Task A2.1: Temperature Logging and Collection (3 points - Mandatory)
In this task, you should record the temperature data from Arduino Nano RP2040.
The frequency of the record is to be at least 1 Hz (1 second). Record the data
for at least 15 minutes (900 data rows or more). To make the temperature vary
a bit, do several "hot blow"s like "huh"s or "blows" on the board or simply
put your finger on the sensor while recording the data. Save them into a CSV.
Remember to submit both CSV and your Python code.

Hint: To turn your logging data into CSV file, there are some third-party
software online that you can use or IDE script which does that.

Note: Your CSV file should look something like this:
Timesteps,Temperature
1,28.1
2,28.8
3,29.0
....
I- Collect the data in a CSV file and submit it with the rest of your results (2pts-Mandatory)

II- Visualize the data with a line graph with two axes: time & temperature with criteria below (1pt-Mandatory)
The color of the line should be orange
Add labels for each axes (Temperature (degrees Celsius), Time(seconds)),
Turn on the grids 
Add legend on the top right corner - temperature
"""

def TaskA2_1_II():
    # Read the CSV file
    df = pd.read_csv('lab2/Data/temperature.csv')
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timesteps'], df['Temperature'], color='orange', label='Temperature')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (degrees Celsius)')
    plt.title('Temperature over Time')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

r"""Task A2.2: Motion Logging - Acc/Gyr and Collection (5 points - Mandatory)
In this task, you should record the motion data from Arduino Nano RP2040.
The frequency of the record is to be at least 2 Hz (0.5 second). Record the
data for at least 10 minutes (1200 rows or more). You should collect both
accelerometer and gyroscope data. Wiggle the Arduino slightly and abruptly (!)
in different directions and angles (Do both angles and directions!) while
collecting the motion, to have some peaks in your data. Save them into a CSV.
Do the wiggles in certain intervals so you get something like the figure below.

Note: Your CSV file should look something like this:
Date, Time, Ax, Ay, Az, Gx, Gy, Gz
2025-02-01, 10:28:30:00, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
2025-02-01, 10:28:30:05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
2025-02-01, 10:28:31:00, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
2025-02-01, 10:28:31:05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
2025-02-01, 10:28:32:00, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
....

Hint: To record date and time you can use RTC module and library, but for the
sake of simplicity, you don't need to use that for this task. You can define
the starting date and time on top of your sketch and use the "Delay" command
to define the data logging intervals.

I- Collect the data in a CSV file and submit it with the rest of your results (2pts-Mandatory)

II- Visualize the data with a line graph with two axes with criteria below (1pt-Mandatory)
plot 6 axis of data
Add labels for each axes (Acceleration (m.sq/s.sq), Time(seconds)),
Turn on the grids 
Add legend on the top right corner - the name of your plots should be Ax, Ay, Az, Gx, Gy, Gz

III- (Manipulate the data) 
With one of the functions mentioned in the instruction above, write a code
which deletes rows of data from your dataset that the acceleration is 0 or
close to 0. This will compress your signals to look something like the right
graphs below in Fig. 9. Visualize the data again with the above criteria in a
line graph. Pay attention that you need to look at the data and define the
threshold accordingly for deleting stationary data. The threshold might be
different for different axes (2pt-Mandatory)
"""

def TaskA2_2_II():
    # Read the CSV file
    df = pd.read_csv('lab2/Data/acceleration.csv')
    print(df.info())
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Ax'], color='red', label='Ax')
    plt.plot(df['Time'], df['Ay'], color='green', label='Ay')
    plt.plot(df['Time'], df['Az'], color='blue', label='Az')
    plt.plot(df['Time'], df['Gx'], color='purple', label='Gx')
    plt.plot(df['Time'], df['Gy'], color='orange', label='Gy')
    plt.plot(df['Time'], df['Gz'], color='yellow', label='Gz')
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m.sq/s.sq)')
    plt.title('Acceleration and Gyroscope over Time')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

def TaskA2_2_III():
    # Read the CSV file
    df = pd.read_csv('lab2/Data/acceleration.csv')

    # Remove rows where acceleration values are close to 0
    df = df[
        (df['Ax'].abs() > 0.02) &
        (df['Ay'].abs() > 0.03) & 
        (df['Az'].abs() > 1)]

    # Normalization
    # df.iloc[:, 2:] = (df.iloc[:, 2:] - df.iloc[:, 2:].min()) / (df.iloc[:, 2:].max() - df.iloc[:, 2:].min())

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Ax'], color='red', label='Ax')
    plt.plot(df['Time'], df['Ay'], color='orange', label='Ay')
    plt.plot(df['Time'], df['Az'], color='yellow', label='Az')
    plt.plot(df['Time'], df['Gx'], color='green', label='Gx')
    plt.plot(df['Time'], df['Gy'], color='blue', label='Gy')
    plt.plot(df['Time'], df['Gz'], color='purple', label='Gz')
    plt.xlabel('Time')
    plt.ylabel('Acceleration and Gyroscope')
    plt.title('Acceleration and Gyroscope over Time')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

r"""Task A2.3: Frozen! (5 points- Mandatory)
This is a dataset of collecting in/out temperature data over 5 months in
 (appx. 97000 data rows): Temperature Readings : IOT DevicesLinks to an
   external site.

I- Visualize the indoor and outdoor temperature in one plot with different
 colors of your choice for the last week (start from the top 02-12-2018 to 08-12-2018). 
 (2 pts-Mandatory)

II- Do these modifications on the dataframe made from the CSV dataset: (3pts Mandatory)

Change the "In" and "Out" text of the "Out\In" column to 1 and 0 respectively.
Separate the date and time in the "noted_date" column, into two separate columns.
Keep only the data of the last day 08-12-2018, and remove the rest of the rows with the appropriate function
Submit the modified CSV and your code together.
"""

def TaskA2_3_I():
    df = pd.read_csv('lab2/Data/IOT-Temperature.csv')

    df['noted_date'] = pd.to_datetime(df['noted_date'], dayfirst=True)

    df = df[(df['noted_date'] >= pd.to_datetime('2018-12-02')) & (df['noted_date'] <= pd.to_datetime('2018-12-08'))]

    plt.figure(figsize=(10, 5))
    plt.plot(df[df['out/in'] == "Out"]['noted_date'], df[df['out/in'] == "Out"]['temp'], color='red', label='Outdoor Temperature')
    plt.plot(df[df['out/in'] == "In"]['noted_date'], df[df['out/in'] == "In"]['temp'], color='blue', label='Indoor Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Indoor and Outdoor Temperature')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

def TaskA2_3_II():
    df = pd.read_csv('lab2/Data/IOT-Temperature.csv')

    df['out/in'] = df['out/in'].apply(lambda x: 1 if x == 'Out' else 0)

    df['noted_date'] = pd.to_datetime(df['noted_date'], dayfirst=True)
    df['date'] = df['noted_date'].dt.date
    df['time'] = df['noted_date'].dt.time
    
    df = df[df['date'] == pd.to_datetime('2018-12-08').date()]

    df = df.drop('noted_date', axis=1)

    df.to_csv('lab2/Data/IOT-Temperature-Modified.csv', index=False)
    
if __name__ == "__main__":
    
    # TaskA2_1_II()

    # TaskA2_2_II()
    # TaskA2_2_III()
    
    # TaskA2_3_I()
    # TaskA2_3_II()

    pass