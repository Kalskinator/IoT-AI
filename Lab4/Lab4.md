# AI and Data Management for IoT-VT25 Lab 4, Audio Assignment (Mandatory) 20 points
## 1. Data Collection for The Assignment
### Your dataset should have 4 various states: 3 words –“Apple”, “Orange”, “Cherry”- and an “unknown” state which no word is said or noise in the background. (You have done this before in your previous semester)
#### 1) The data should be collected via Arduino built-in microphone.
#### 2) Extract audio files as Wav format and create a label file for them (Like csv orexcel file)
#### 3) You should at least have 12 min of input data (3 min for each state. For example20 samplesfor each word)
#### 4) Use 80% of the data for training and 20% for testing.

[Lab4_Mic_Data_Train_Test_Split](/aleks_lab4_mic-export_train_test_split.zip)


## 2. Model and Training the network
#### 1) In the code file, there is a model (M5) for training, which is a simple CNN model. If you choose to use this model, there is no need to convert the audio files into other formats such as spectrograms. However, if you want to use a more complex CNN model like Wav2Vec 2 (also available in the code file), you should first convert the audio files into another format (e.g., spectrograms).

```python
import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as AF
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchaudio
from torchaudio import transforms as T
import torchaudio.functional as AF

from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from IPython.display import Audio


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomSpeechDataset(Dataset):
    def __init__(self, root_dir, json_path, subset="training"):
        # Load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)


        self.files = [item for item in data["files"] if item["category"] == subset]
        self.root_dir = root_dir

        # Create label to index mapping
        unique_labels = sorted(set(item["label"]["label"] for item in self.files))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = self.files[idx]
        waveform, sample_rate = torchaudio.load(f"{self.root_dir}/{file_data['path']}")
        label = self.label_to_idx[file_data["label"]["label"]]
        return waveform, sample_rate, label


root_dir = "./Lab4/Data"
json_path = "./Lab4/Data/info.labels"

NUM_OF_CLASSES = 4
EPOCHS = 50
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = CustomSpeechDataset(root_dir, json_path, subset="training")
test_set = CustomSpeechDataset(root_dir, json_path, subset="testing")

labels = set([d[2] for d in train_set])
label2num = {label: num for num, label in enumerate(labels)}


def collate_fn(batch):
    data = [b[0][0] for b in batch]
    data = pad_sequence(data, batch_first=True)
    data = AF.resample(data, 16000, 8000).unsqueeze(1)
    labels = torch.LongTensor([label2num[b[2]] for b in batch])
    return data, labels


def train_one_epoch(model, train_loader, sloss_fn, optimizer, epoch=None):
    model.train()
    loss_train = AverageMeter()
    acc_train = Accuracy(task="multiclass", num_classes=NUM_OF_CLASSES).to(device)
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            acc_train(outputs, targets.int())
            tepoch.set_postfix(loss=loss_train.avg, accuracy=100.0 * acc_train.compute().item())
    return model, loss_train.avg, acc_train.compute().item()


def validation(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        acc_valid = Accuracy(task="multiclass", num_classes=NUM_OF_CLASSES).to(device)
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss_valid.update(loss.item())
            acc_valid(outputs, targets.int())
    return loss_valid.avg, acc_valid.compute().item()


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Number of training samples: {len(train_set)}")
print(f"Number of testing samples: {len(test_set)}")
print(f"Available labels: {list(train_set.label_to_idx.keys())}")


model = M5(n_input=1, n_output=NUM_OF_CLASSES).to(device)
loss_fn = nn.CrossEntropyLoss()


lr = 0.03
wd = 1e-5
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

num_epochs = 20

for epoch in range(num_epochs):
    # Train
    model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
    # Validation
    loss_valid, acc_valid = validation(model, test_loader, loss_fn)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    acc_train_hist.append(acc_train)
    acc_valid_hist.append(acc_valid)

    if loss_valid < best_loss_valid:
        torch.save(model, f"Lab4/Models/model.pt")
        best_loss_valid = loss_valid
        print("Model Saved!")

    print(f"Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}")
    print()

    epoch_counter += 1


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epoch_counter), loss_train_hist, "r-", label="Train")
plt.plot(range(epoch_counter), loss_valid_hist, "b-", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.title("Training and Validation Loss")


plt.subplot(1, 2, 2)
plt.plot(range(epoch_counter), acc_train_hist, "r-", label="Train")
plt.plot(range(epoch_counter), acc_valid_hist, "b-", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.title("Training and Validation Accuracy")


plt.tight_layout()

plt.show()

```

#### 2) Train the model for at least 50 epochs and report the accuracy.

![Training Curves](/Lab4/Figures/training_curves.png)

#### 3) (Optional) If you want to learn more, you can modify the M5 model by adding additional layers, train the modified model, and then compare its performance with the original M5 model.

# AI and Data Management for IoT-VT25 Lab 4, CSV or Image Assignment (Optional, 10 points)
## This task is optional. You will get 10 points if you complete it (CSV or Image). The entire structure has been explained in the PowerPoint file. Feel free to ask any questions.
```python
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


training_data = torchvision.datasets.CIFAR10(
    root="./Lab4/data", train=True, transform=transforms.ToTensor()
)
test_data = torchvision.datasets.CIFAR10(
    root="./Lab4/data", train=False, transform=transforms.ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch}", f" number of images:{i}", "loss: ", running_loss)

correct = 0
total = 0

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

torch.save(net.state_dict(), "./Lab4/Models/cifar_net.pth")

```
62.31% Accuracy