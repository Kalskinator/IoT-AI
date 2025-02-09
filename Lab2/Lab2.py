import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def TaskA2_1_II():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/temperature.csv")
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(
        df["Timesteps"], df["Temperature"], color="orange", label="Temperature"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (degrees Celsius)")
    plt.title("Temperature over Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_2_II():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/acceleration.csv")
    print(df.info())
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Ax"], color="red", label="Ax")
    plt.plot(df["Time"], df["Ay"], color="green", label="Ay")
    plt.plot(df["Time"], df["Az"], color="blue", label="Az")
    plt.plot(df["Time"], df["Gx"], color="purple", label="Gx")
    plt.plot(df["Time"], df["Gy"], color="orange", label="Gy")
    plt.plot(df["Time"], df["Gz"], color="yellow", label="Gz")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m.sq/s.sq)")
    plt.title("Acceleration and Gyroscope over Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_2_III():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/acceleration.csv")

    # Remove rows where acceleration values are close to 0
    df = df[
        (df["Ax"].abs() > 0.02) & (df["Ay"].abs() > 0.03) & (df["Az"].abs() > 1)
    ]

    # Normalization
    # df.iloc[:, 2:] = (df.iloc[:, 2:] - df.iloc[:, 2:].min()) / (df.iloc[:, 2:].max() - df.iloc[:, 2:].min())

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Ax"], color="red", label="Ax")
    plt.plot(df["Time"], df["Ay"], color="orange", label="Ay")
    plt.plot(df["Time"], df["Az"], color="yellow", label="Az")
    plt.plot(df["Time"], df["Gx"], color="green", label="Gx")
    plt.plot(df["Time"], df["Gy"], color="blue", label="Gy")
    plt.plot(df["Time"], df["Gz"], color="purple", label="Gz")
    plt.xlabel("Time")
    plt.ylabel("Acceleration and Gyroscope")
    plt.title("Acceleration and Gyroscope over Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_3_I():
    df = pd.read_csv("lab2/Data/IOT-Temperature.csv")

    df["noted_date"] = pd.to_datetime(df["noted_date"], dayfirst=True)

    df = df[
        (df["noted_date"] >= pd.to_datetime("2018-12-02"))
        & (df["noted_date"] <= pd.to_datetime("2018-12-08"))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(
        df[df["out/in"] == "Out"]["noted_date"],
        df[df["out/in"] == "Out"]["temp"],
        color="red",
        label="Outdoor Temperature",
    )
    plt.plot(
        df[df["out/in"] == "In"]["noted_date"],
        df[df["out/in"] == "In"]["temp"],
        color="blue",
        label="Indoor Temperature",
    )
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Indoor and Outdoor Temperature")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_3_II():
    df = pd.read_csv("lab2/Data/IOT-Temperature.csv")

    df["out/in"] = df["out/in"].apply(lambda x: 1 if x == "Out" else 0)

    df["noted_date"] = pd.to_datetime(df["noted_date"], dayfirst=True)
    df["date"] = df["noted_date"].dt.date
    df["time"] = df["noted_date"].dt.time

    df = df[df["date"] == pd.to_datetime("2018-12-08").date()]

    df = df.drop("noted_date", axis=1)

    df.to_csv("lab2/Data/IOT-Temperature-Modified.csv", index=False)


if __name__ == "__main__":

    # TaskA2_1_II()

    # TaskA2_2_II()
    # TaskA2_2_III()

    # TaskA2_3_I()
    # TaskA2_3_II()

    pass
