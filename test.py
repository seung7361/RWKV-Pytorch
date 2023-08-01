import torch

time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
x = torch.randint(low=0, high=10, size=(1, 5, 5))
print(x)
print(time_shift(x))