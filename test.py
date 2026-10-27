from collections import Counter
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

colors = ["red", "blue", "red", "green", "red", "blue"]
print(Counter(colors))

print(Counter(train_data.targets.tolist()))