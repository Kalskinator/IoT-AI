import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = torch.sigmoid(self.linear(x))
        return x

class TaskA3:

    def load_data():
        train_dataset = datasets.MNIST(root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root="./Lab3/Data", train=False, transform=transforms.ToTensor())
        train_data = train_dataset.data[:10000].view(-1, 28 * 28).numpy()
        train_labels = train_dataset.targets[:10000].numpy()
        test_data = test_dataset.data.view(-1, 28 * 28).numpy()
        test_labels = test_dataset.targets.numpy()
        return train_data, train_labels, test_data, test_labels

    def linear_regression():
        train_dataset = datasets.MNIST(root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root="./Lab3/Data", train=False, transform=transforms.ToTensor())

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
        optimizer = optim.Adam(linearRegression.parameters(), LEARNING_RATE)  # defining the optimizer

        lr_mse = []
        for epoch in range(EPOCHS):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()
                # get output from the model, given the inputs
                outputs = linearRegression(images)

                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                # get loss for the predicted output
                loss = loss_function(outputs, labels_one_hot)
                # get gradients w.r.t to parameters
                loss.backward()
                # update parameters
                optimizer.step()
            lr_mse.append(loss.item())
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = linearRegression(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print("Epoch: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy))
        return lr_mse

    def support_vector_machine():
        train_data, train_labels, test_data, test_labels = TaskA3.load_data()

        svm_classifier = svm.SVC(kernel="linear")
        svm_classifier.fit(train_data, train_labels)
        svm_predictions = svm_classifier.predict(test_data)
        svm_accuracy = accuracy_score(test_labels, svm_predictions)
        mse = mean_squared_error(test_labels, svm_predictions)
        print("SVM Accuracy: {}".format(svm_accuracy))
        return [mse] * 10

    def random_forest():
        train_data, train_labels, test_data, test_labels = TaskA3.load_data()
        rf_classifier = RandomForestClassifier(max_depth=10)
        rf_classifier.fit(train_data, train_labels)
        rf_predictions = rf_classifier.predict(test_data)
        rf_accuracy = accuracy_score(test_labels, rf_predictions)
        mse = mean_squared_error(test_labels,rf_predictions)
        print("Random Forest Accuracy: {}".format(rf_accuracy))
        return [mse] * 10

    def visualize_errors(lr_mse, svm_mse, rf_mse):
        epochs = range(1, len(lr_mse) + 1)
        plt.plot(epochs, lr_mse, 'r', label='LR')
        plt.plot(epochs, svm_mse, 'g', label='SVM')
        plt.plot(epochs, rf_mse, 'b', label='RF')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.title('MSE vs Epoch')
        plt.show()

if __name__ == "__main__":
    lr_mse = TaskA3.linear_regression()
    svm_mse = TaskA3.support_vector_machine()
    rf_mse = TaskA3.random_forest()
    TaskA3.visualize_errors(lr_mse, svm_mse, rf_mse)