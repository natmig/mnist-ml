import torch
from torch.utils.data import random_split

# Toy example of splitting a list of numbers into two parts
# Providing a seed so the list of numbers stays the same every time we run the code
# since otherwise random_split would cause print(list(part_b)) to be diffent each time
numbers = list((range(10)))
gen = torch.Generator().manual_seed(42)
part_a, part_b = random_split(numbers, [7,3], generator=gen)
print(list(part_b))