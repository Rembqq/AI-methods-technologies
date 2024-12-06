import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


def x_to_y(x):
    return np.where(x == 0, 1, np.sin(x) / x)


def func(x, y):
    return (x - y) * np.sin(x + y)


x = np.linspace(0, 1, 50)

z = func(x, x_to_y(x))
input_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
output_tensor = torch.tensor(z.reshape(-1, 1), dtype=torch.float32)


class Cascade20(nn.Module):
    def __init__(self):
        super(Cascade20, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Cascade1010(nn.Module):
    def __init__(self):
        super(Cascade1010, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Feed10(nn.Module):
    def __init__(self):
        super(Feed10, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Feed20(nn.Module):
    def __init__(self):
        super(Feed20, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Elman15(nn.Module):
    def __init__(self):
        super(Elman15, self).__init__()
        self.fc1 = nn.Linear(1, 15)
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Elman555(nn.Module):
    def __init__(self):
        super(Elman555, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_and_evaluate(network, epochs=5000):
    net = network()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs_pred = net(input_tensor)
        loss = criterion(outputs_pred, output_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f'Generation: {epoch}, Loss: {loss.item()}')

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot the original and approximated functions
    z_pred = net(input_tensor).detach().numpy().squeeze()
    ax1.plot(x, z, label='Real Function', color='orange')
    ax1.plot(x, z_pred, label='Approximated Function', color='blue')
    ax1.set_title(f'Real and Approximated Functions for {network.__name__}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.legend()

    # Plot the training loss
    ax2.plot(range(epochs), losses, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{network.__name__} | Loss vs Epoch')

    plt.show()


# Train and evaluate the networks
train_and_evaluate(Feed10)
train_and_evaluate(Feed20)
train_and_evaluate(Cascade20)
train_and_evaluate(Cascade1010)
train_and_evaluate(Elman15)
train_and_evaluate(Elman555)
