import torch
from torchvision import datasets # Allows MNIST dataset
from torchvision.transforms import ToTensor # Allows transformation to tensor
from sklearn.linear_model import LogisticRegression # Allows the use of logistic regression
from torch.utils.data import random_split # Allows splitting of data
from collections import Counter # Allows counting

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# Split data into two parts, but have a seed so the data stays consistent each run.
gen = torch.Generator().manual_seed(42)
training_data, validation_data = random_split(train_data, [50000, 10000], generator=gen)
print(Counter(train_data.targets.tolist()[i] for i in validation_data.indices))
X_all = train_data.data.reshape(60000,784) / 255
X_train = X_all[training_data.indices]
y_train = train_data.targets[training_data.indices]
X_val = X_all[validation_data.indices]
y_val = train_data.targets[validation_data.indices]
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Fits the model to the data and evaluates accuracy on unseen data.
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model accuracy on validation set:", model.score(X_val, y_val))
print("Model accuracy on training set:", model.score(X_train, y_train))