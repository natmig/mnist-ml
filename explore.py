import matplotlib.pyplot as plt # Allows for visualization 
from torchvision import datasets # Allows me to use MNIST dataset
from torchvision.transforms import ToTensor # Transforms to tensor
import random # To use for random selection of images when outputing
from torch.utils.data import random_split
import torch

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

print("Number of images:", len(train_data))

image, label = train_data[0]
[1, 28, 28]
print("Image shape:", image.shape)
print("Correct Label:", label)
print(image.min(), image.max())

# Split data into two parts, but have a seed so the data stays consistent each run.
gen = torch.generator().manual_seed(42)
training_data, validation_data = random_split(train_data, 50000, 10000, generator=gen)

# Creates a row with 10 subplots, each showing a random image from the training dataset
fig,axes = plt.subplots(1,10, figsize= (12, 3))
for i in range(10):
    idx = random.randrange(len(train_data))
    image, label = train_data[idx]
    axes[i].imshow(image.squeeze(), cmap="gray")
    axes[i].set_title(f"Digit: {label}")
    axes[i].axis("off")
plt.show()