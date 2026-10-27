import matplotlib.pyplot as plt

fig,axes = plt.subplots(1,3, figsize=(12, 3))
for i in range(3):
    axes[i].plot([0, 1, 2], [0, i, 0])
    axes[i].set_title(f"Plot {i}")
plt.show()