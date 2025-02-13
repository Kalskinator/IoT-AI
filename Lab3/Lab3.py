import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = torch.sigmoid(self.linear(x))
        return x


def Task_A3_I():

    # BATCH_SIZE = 32
    # N_INPUTS = 28 * 28
    # N_OUTPUTS = 10
    # LEARNING_RATE = 0.01
    # EPOCHS = 50

    # Loading training data
    train_dataset = datasets.MNIST(
        root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True
    )
    # Loading test data
    test_dataset = datasets.MNIST(root="./Lab3/Data", train=False, transform=transforms.ToTensor())

    # train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # # Initialize the model, loss function, and optimizer
    # linearRegression = LinearRegression(N_INPUTS, N_OUTPUTS)

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(linearRegression.parameters(), LEARNING_RATE)  # defining the optimizer

    # acc = []
    # for epoch in range(EPOCHS):
    #     for images, labels in train_loader:
    #         # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    #         optimizer.zero_grad()
    #         # get output from the model, given the inputs
    #         outputs = linearRegression(images)
    #         # get loss for the predicted output
    #         loss = loss_function(outputs, labels)
    #         # get gradients w.r.t to parameters
    #         loss.backward()
    #         # update parameters
    #         optimizer.step()

    #     correct = 0
    #     for images, labels in test_loader:
    #         outputs = linearRegression(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         correct += (predicted == labels).sum()
    #     accuracy = 100 * (correct.item()) / len(test_dataset)
    #     acc.append(accuracy)
    #     print("Epoch: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy))

    # SVM Classifier
    svm_classifier = svm.SVC(kernel="linear", verbose=True)
    train_data = train_dataset.data.view(-1, 28 * 28).numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.view(-1, 28 * 28).numpy()
    test_labels = test_dataset.targets.numpy()

    svm_classifier.fit(train_data, train_labels)
    svm_predictions = svm_classifier.predict(test_data)
    svm_accuracy = accuracy_score(test_labels, svm_predictions)
    print("SVM Accuracy: {}".format(svm_accuracy))

    # # Random Forest Classifier
    # rf_classifier = RandomForestClassifier(max_depth=10)
    # rf_classifier.fit(train_data, train_labels)
    # rf_predictions = rf_classifier.predict(test_data)
    # rf_accuracy = accuracy_score(test_labels, rf_predictions)
    # print("Random Forest Accuracy: {}".format(rf_accuracy))


if __name__ == "__main__":
    Task_A3_I()
