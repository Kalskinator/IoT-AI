import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


class ModelTrainer:
    @staticmethod
    def support_vector_machine(load_data_func):
        train_data, train_labels, test_data, test_labels = load_data_func()
        model = svm.SVC(kernel="linear")
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        print(f"SVM Accuracy: {accuracy*100:.2f}%")
        return [mse] * 10

    @staticmethod
    def random_forest(load_data_func):
        train_data, train_labels, test_data, test_labels = load_data_func()
        model = RandomForestClassifier(max_depth=10)
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        print(f"Random Forest Accuracy: {accuracy*100:.2f}%")
        return [mse] * 10

    @staticmethod
    def visualize_errors(lr_mse, svm_mse, rf_mse):
        epochs = range(1, len(lr_mse) + 1)
        plt.plot(epochs, lr_mse, "r", label="LR")
        plt.plot(epochs, svm_mse, "g", label="SVM")
        plt.plot(epochs, rf_mse, "b", label="RF")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend(loc="upper right")
        plt.title("MSE vs Epoch")
        plt.show()


class TaskA3_I:

    def load_data():
        train_dataset = datasets.MNIST(
            root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = datasets.MNIST(
            root="./Lab3/Data", train=False, transform=transforms.ToTensor()
        )
        train_data = train_dataset.data[:10000].view(-1, 28 * 28).numpy()
        train_labels = train_dataset.targets[:10000].numpy()
        test_data = test_dataset.data.view(-1, 28 * 28).numpy()
        test_labels = test_dataset.targets.numpy()
        return train_data, train_labels, test_data, test_labels

    def linear_regression():
        train_dataset = datasets.MNIST(
            root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = datasets.MNIST(
            root="./Lab3/Data", train=False, transform=transforms.ToTensor()
        )

        BATCH_SIZE = 32
        N_INPUTS = 28 * 28
        N_OUTPUTS = 10
        LEARNING_RATE = 0.01
        EPOCHS = 10

        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

        # Initialize the model, loss function, and optimizer
        linearRegression = LinearRegression(N_INPUTS, N_OUTPUTS)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(
            linearRegression.parameters(), LEARNING_RATE
        )  # defining the optimizer

        mse_list = []
        for epoch in range(EPOCHS):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()
                # get output from the model, given the inputs
                outputs = linearRegression(images.view(-1, 28 * 28))

                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                # get loss for the predicted output
                loss = loss_function(outputs, labels_one_hot)
                # get gradients w.r.t to parameters
                loss.backward()
                # update parameters
                optimizer.step()
            mse_list.append(loss.item())
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = linearRegression(images.view(-1, 28 * 28))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print("Epoch: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy))
        return mse_list

    def support_vector_machine():
        return ModelTrainer.support_vector_machine(TaskA3_I.load_data)

    def random_forest():
        return ModelTrainer.random_forest(TaskA3_I.load_data)

    def visualize_errors(lr_mse, svm_mse, rf_mse):
        ModelTrainer.visualize_errors(lr_mse, svm_mse, rf_mse)


class TaskA3_II:
    def load_data():
        df = pd.read_csv("./Lab3/Data/seattle-weather.csv")
        df = df.dropna()

        df["weather"] = df["weather"].map({"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4})

        X = df[["precipitation", "temp_max", "temp_min", "wind"]]
        y = df["weather"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def linear_regression():
        X_train, X_test, y_train, y_test = TaskA3_II.load_data()
        INPUT_SIZE = X_train.shape[1]
        OUTPUT_SIZE = len(
            pd.get_dummies(y_train).columns
        )  # Ensure output size matches number of classes
        LEARNING_RATE = 0.01
        EPOCHS = 10
        model = LinearRegression(INPUT_SIZE, OUTPUT_SIZE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(pd.get_dummies(y_train).values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(pd.get_dummies(y_test).values, dtype=torch.float32)
        mse_list = []

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            mse_list.append(loss.item())
            print(f"Epoch: {epoch}")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            mse = criterion(outputs, y_test_tensor).item()
            mse_list.append(mse)

        return mse_list

    def support_vector_machine():
        X_train, X_test, y_train, y_test = TaskA3_II.load_data()
        return ModelTrainer.support_vector_machine(lambda: (X_train, y_train, X_test, y_test))

    def random_forest():
        X_train, X_test, y_train, y_test = TaskA3_II.load_data()
        return ModelTrainer.random_forest(lambda: (X_train, y_train, X_test, y_test))

    def visualize_errors(lr_mse, svm_mse, rf_mse):
        ModelTrainer.visualize_errors(lr_mse, svm_mse, rf_mse)


class TaskA3_IV:

    def k_means():

        df = pd.read_csv(
            "./Lab3/Data/penguins.csv", usecols=["species", "bill_length_mm", "bill_depth_mm"]
        )

        df = df.dropna()

        X = df[["bill_length_mm", "bill_depth_mm"]]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        inertias = []
        # testing the data on 1 to 11 clusters and calculating the inertia
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertias, marker="o")
        plt.title("Elbow method")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.show()

        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
        # kmeans = KMeans(n_clusters=3)
        df["cluster"] = kmeans.fit_predict(X)

        # Map cluster labels to species names
        cluster_to_species = {0: "Adelie", 2: "Chinstrap", 1: "Gentoo"}
        df["species_cluster"] = df["cluster"].map(lambda x: cluster_to_species[x])

        # Visualizing the clusters
        for cluster, color in zip(cluster_to_species.values(), ["royalblue", "orange", "green"]):
            cluster_data = df[df["species_cluster"] == cluster]
            plt.scatter(
                cluster_data["bill_length_mm"],
                cluster_data["bill_depth_mm"],
                label=cluster,
                color=color,
            )

        plt.xlabel("Bill Length (mm)")
        plt.ylabel("Bill Depth (mm)")
        plt.title("K-Means Clustering of Penguins")
        plt.legend(loc="upper right")
        plt.show()

        # Inverse transform the centroids back to the original scale
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        for cluster, color in zip(cluster_to_species.values(), ["royalblue", "orange", "green"]):
            cluster_data = df[df["species_cluster"] == cluster]
            plt.scatter(
                cluster_data["bill_length_mm"],
                cluster_data["bill_depth_mm"],
                label=cluster,
                color=color,
            )
        plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", label="Centroids")
        plt.xlabel("Bill Length (mm)")
        plt.ylabel("Bill Depth (mm)")
        plt.title("K-Means Clustering of Penguins with Centroids")
        plt.legend(loc="upper right")
        plt.show()

        # Evaluate the model
        correct_labels = sum(df["species"] == df["species_cluster"])
        accuracy = correct_labels / len(df)
        print(f"K-Means Clustering Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    # mnist_lr_mse = TaskA3_I.linear_regression()
    # mnist_svm_mse = TaskA3_I.support_vector_machine()
    # mnist_rf_mse = TaskA3_I.random_forest()
    # TaskA3_I.visualize_errors(mnist_lr_mse, mnist_svm_mse, mnist_rf_mse)

    weather_lr_mse = TaskA3_II.linear_regression()
    weather_svm_mse = TaskA3_II.support_vector_machine()
    weather_rf_mse = TaskA3_II.random_forest()
    TaskA3_II.visualize_errors(weather_lr_mse, weather_svm_mse, weather_rf_mse)

    # TaskA3_IV.k_means()
