# question: https://stackoverflow.com/questions/74171994/diagonal-or-divergence-of-jacobian-matrix/75467342#75467342
# solution from ChatGPT
import torch


# Define the function for which we want to compute the Jacobian
def func(x):
    y1 = x[0] ** 2 + x[1] ** 3
    y2 = x[1] ** 2 + x[0] ** 3
    return torch.stack([y1, y2])

# Define the input at which we want to evaluate the Jacobian
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Compute the Jacobian of the function with respect to the input
J = torch.autograd.functional.jacobian(func, x)

print(J)
