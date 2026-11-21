import matplotlib.pyplot as plt # Allows for visualization 
from torchvision import datasets # Allows me to use MNIST dataset
from torchvision.transforms import ToTensor # Transforms to tensor
import torch
import random

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


# Creates a row with 10 subplots, each showing a random image from the training dataset
fig,axes = plt.subplots(1,10, figsize= (12, 3))
for i in range(10):
    idx = random.randrange(len(train_data))
    image, label = train_data[idx]
    axes[i].imshow(image.squeeze(), cmap="gray")
    axes[i].set_title(f"Digit: {label}")
    axes[i].axis("off")
plt.show()
