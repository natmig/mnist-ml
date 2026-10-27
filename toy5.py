import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Toy example of splitting a list of numbers into two parts
numbers = list(range(10))
part_a, part_b = random_split(numbers, [7, 3])
print(len(part_a), len(part_b))


# Split the training data into two parts, 
training_data, validation_data = random_split(train_data, [50000, 10000])
print(len(training_data), len(validation_data))
print(list(training_data)[:5])
print(list(validation_data)[:5])