from collections import Counter
import torch

labels = torch.tensor([3, 1, 3, 2, 3, 1])
print(Counter(labels.tolist()))