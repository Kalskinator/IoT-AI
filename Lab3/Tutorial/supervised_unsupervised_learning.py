# To load the whole library for making datasets, torch optimizer, and other functions
import torch
import pandas as pd

# to use torch functions related to building neural networks, fitting data, loss functions and other functions
import torch.nn as nn


# making tensor input and output data
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# print(x_data)
# print(y_data)


# load the dataset from pandas data frame - X and Y are pandas data frames
# data = {"Height": [165.4, 175.9, 125.2, 189.5], "Age": [25, 30, 15, 40]}
# df = pd.DataFrame(data)
# X = df["Height"]
# Y = df["Age"]
# X = torch.tensor(X, dtype=torch.float32)
# Y = torch.tensor(Y, dtype=torch.int)
# print(X)
# print(Y)

# Reshape function used to transpose or lower the dimension of the
# data- if your y function consists of 1 column of data in the shape of (1,X),
# you can reduce the number of dimensions to one by the below command.
# Y = torch.tensor(Y, dtype=torch.int).reshape(-1, 1)
# print(Y)

# For saving and loading your trained model in Pytorch the below codes are used. "model" is your trained model:

# Showing learn parameters during the training stage

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# SAving your model

torch.save(model.state_dict(), PATH)

# LOading your model

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
