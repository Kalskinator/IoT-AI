import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def TaskA2_1_II():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/temperature.csv")
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Timesteps"], df["Temperature"], color="orange", label="Temperature")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (degrees Celsius)")
    plt.title("Temperature over Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_2_II():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/acceleration.csv")
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Ax"], color="red", label="Ax")
    plt.plot(df["Time"], df["Ay"], color="orange", label="Ay")
    plt.plot(df["Time"], df["Az"], color="yellow", label="Az")
    plt.plot(df["Time"], df["Gx"], color="green", label="Gx")
    plt.plot(df["Time"], df["Gy"], color="blue", label="Gy")
    plt.plot(df["Time"], df["Gz"], color="purple", label="Gz")
    plt.xlabel("Time(seconds)")
    plt.ylabel("Acceleration (m.sq/s.sq) and Gyroscope (rad/s)")
    plt.title("Acceleration and Gyroscope over Time")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def TaskA2_2_III():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/acceleration.csv")

    # Remove rows where acceleration values are close to 0
    df = df[(df["Ax"].abs() > 0.02) & (df["Ay"].abs() > 0.03) & (df["Az"].abs() > 1)]

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
    plt.xlabel("Time(seconds)")
    plt.ylabel("Acceleration (m.sq/s.sq) and Gyroscope (rad/s)")
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


def TaskA2_4_I():
    df = pd.read_csv("lab2/Data/aw_fb_data.csv")

    df["log"] = np.log(df["calories"])
    df["log"].hist()

    plt.show()


def plot_participant_data(ax, x, y_data, labels, colors, title, xlabel, ylabel):
    ax.stackplot(x, *y_data, labels=labels, colors=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center")
    ax.grid(True)


def TaskA2_4_II():
    df = pd.read_csv("lab2/Data/aw_fb_data.csv")
    df_copy = df.groupby(["age", "height", "weight"]).first().reset_index()

    _, axs = plt.subplots(2, 2, figsize=(10, 8))

    plot_participant_data(
        axs[0, 0],
        df_copy.index,
        [df_copy["age"]],
        ["Age"],
        ["red"],
        "Age of Participants",
        "Participant",
        "Age",
    )
    plot_participant_data(
        axs[0, 1],
        df_copy.index,
        [df_copy["height"]],
        ["Height"],
        ["green"],
        "Height of Participants",
        "Participant",
        "Height (cm)",
    )
    plot_participant_data(
        axs[1, 0],
        df_copy.index,
        [df_copy["weight"]],
        ["Weight"],
        ["blue"],
        "Weight of Participants",
        "Participant",
        "Weight (kg)",
    )
    plot_participant_data(
        axs[1, 1],
        np.arange(len(df_copy)),
        [df_copy["age"], df_copy["height"], df_copy["weight"]],
        ["Age", "Height", "Weight"],
        ["red", "green", "blue"],
        "Age, Height, and Weight of Participants",
        "Participant",
        "Age, Height, and Weight",
    )

    plt.tight_layout()
    plt.show()


def pad_and_stackplot(ax, data, max_len, labels, colors, title, ylabel):
    padded_data = [
        np.pad(participant_data, (0, max_len - len(participant_data))) for participant_data in data
    ]
    ax.stackplot(np.arange(max_len), padded_data, labels=labels, colors=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True)


def TaskA2_4_III():
    # Read the CSV file
    df = pd.read_csv("lab2/Data/aw_fb_data.csv")

    df_unique = pd.read_csv("lab2/Data/aw_fb_data_Unique.csv")
    df_unique["id"] = range(1, len(df_unique) + 1)
    unique_id_cols = ["age", "height", "weight", "gender"]
    df_merged = df.merge(df_unique[unique_id_cols + ["id"]], on=unique_id_cols, how="left")

    id1, id2, id3 = 6, 28, 36

    participant_1 = df_merged[df_merged["id"] == id1].copy()
    participant_2 = df_merged[df_merged["id"] == id2].copy()
    participant_3 = df_merged[df_merged["id"] == id3].copy()

    max_len = max(len(participant_1), len(participant_2), len(participant_3))

    _, axs = plt.subplots(3, 1, figsize=(15, 8))

    pad_and_stackplot(
        axs[0],
        [participant_1["steps"], participant_2["steps"], participant_3["steps"]],
        max_len,
        labels=[f"Participant #{id1}", f"Participant #{id2}", f"Participant #{id3}"],
        colors=["red", "green", "blue"],
        title="Steps of First Three Participants",
        ylabel="Steps",
    )

    pad_and_stackplot(
        axs[1],
        [participant_1["hear_rate"], participant_2["hear_rate"], participant_3["hear_rate"]],
        max_len,
        labels=[f"Participant #{id1}", f"Participant #{id2}", f"Participant #{id3}"],
        colors=["red", "green", "blue"],
        title="Heart Rate of First Three Participants",
        ylabel="Heart Rate",
    )

    pad_and_stackplot(
        axs[2],
        [participant_1["calories"], participant_2["calories"], participant_3["calories"]],
        max_len,
        labels=[f"Participant #{id1}", f"Participant #{id2}", f"Participant #{id3}"],
        colors=["red", "green", "blue"],
        title="Calories of First Three Participants",
        ylabel="Calories",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # TaskA2_1_II()

    # TaskA2_2_II()
    # TaskA2_2_III()

    # TaskA2_3_I()
    # TaskA2_3_II()

    # TaskA2_4_I()
    # TaskA2_4_II()
    TaskA2_4_III()

    pass
