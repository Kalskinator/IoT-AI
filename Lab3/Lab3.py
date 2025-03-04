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
from sklearn.metrics import confusion_matrix
import seaborn as sns


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


class ModelTrainer:
    @staticmethod
    def support_vector_machine(train_data, train_labels, test_data, test_labels):
        model = svm.SVC(kernel="linear")
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        print(f"SVM Accuracy: {accuracy*100:.2f}%")
        return mse

    @staticmethod
    def random_forest(train_data, train_labels, test_data, test_labels):
        model = RandomForestClassifier(max_depth=10)
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        mse = mean_squared_error(test_labels, predictions)
        print(f"Random Forest Accuracy: {accuracy*100:.2f}%")
        return mse

    @staticmethod
    def visualize_errors(lr_mse, svm_mse, rf_mse):
        epochs = range(1, len(lr_mse) + 1)
        plt.plot(epochs, lr_mse, "r", label="LR")
        plt.plot(epochs, [svm_mse] * len(lr_mse), "g", label="SVM")
        plt.plot(epochs, [rf_mse] * len(lr_mse), "b", label="RF")
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
        model = LinearRegression(N_INPUTS, N_OUTPUTS)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)  # defining the optimizer

        mse_list = []
        for epoch in range(EPOCHS):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()
                # get output from the model, given the inputs
                outputs = model(images.view(-1, 28 * 28))

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
                outputs = model(images.view(-1, 28 * 28))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print("Epoch: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy))
        return mse_list

    def support_vector_machine():
        return ModelTrainer.support_vector_machine(*TaskA3_I.load_data())

    def random_forest():
        return ModelTrainer.random_forest(*TaskA3_I.load_data())


class TaskA3_II:

    def load_data():
        df = pd.read_csv("./Lab3/Data/seattle-weather.csv")
        df = df.dropna()
        df = df.drop(columns=["date"])  # Drop the date column
        X = df.drop(columns=["weather"])
        y = df["weather"].map({"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, y_train, X_test, y_test

    def linear_regression():
        X_train, y_train, X_test, y_test = TaskA3_II.load_data()

        EPOCHS = 10000
        N_INPUTS = X_train.shape[1]
        N_OUPUTS = 5
        LEARNING_RATE = 0.01

        model = LinearRegression(N_INPUTS, N_OUPUTS)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train.values, dtype=torch.long)
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test.values, dtype=torch.long)

        mse_list = []
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            labels_one_hot = nn.functional.one_hot(labels, num_classes=5).float()
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()
            mse_list.append(loss.item())

            # Print epoch loss
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                _, predicted = torch.max(test_outputs.data, 1)
                cm = confusion_matrix(test_labels, predicted)
                accuracy = (predicted == test_labels).sum().item() / len(y_test)
                print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {accuracy*100:.2f}%")

        # Visualize the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["drizzle", "rain", "sun", "snow", "fog"],
            yticklabels=["drizzle", "rain", "sun", "snow", "fog"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix for Linear Regression")
        plt.show()

        return mse_list

    def support_vector_machine():
        return ModelTrainer.support_vector_machine(*TaskA3_II.load_data())

    def random_forest():
        return ModelTrainer.random_forest(*TaskA3_II.load_data())


class TaskA3_III:
    pass


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
    # ModelTrainer.visualize_errors(
    #     TaskA3_I.linear_regression(), TaskA3_I.support_vector_machine(), TaskA3_I.random_forest()
    # )

    # ModelTrainer.visualize_errors(
    #     TaskA3_II.linear_regression(), TaskA3_II.support_vector_machine(), TaskA3_II.random_forest()
    # )

    TaskA3_III

    # TaskA3_IV.k_means()
