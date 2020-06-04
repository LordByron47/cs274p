import numpy as np
import torch
from torch import nn
import math
from pprint import pprint
from tqdm.notebook import tqdm

class DeepNeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, layer_depth=4, activation=nn.ReLU):
        super(DeepNeuralNetwork, self).__init__()

        self.activation = activation()
        self.in_size = in_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcn = nn.ModuleDict({})

        for l in range(layer_depth):
            name = 'fc'+str(1+l)
            self.fcn[name] = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, out_size)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.loss_tracker = []

    def add_layer(index, layer):
        pass

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for k, l in self.fcn.items():
            x = self.activation(x)
        x = self.out(x)
        return x

    def train(self, inputs, labels, test_inputs=None, test_labels=None, epochs=10) -> None:
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            outputs = self(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            self.loss_tracker.append(loss.item())
            acc = self.accuracy(test_inputs, test_labels)
            tqdm.write("Epoch {}, Loss: {} Acc: {}".format(epoch, loss.item(), acc))

    def accuracy(self, test_inputs, test_labels):
        _, preds_y = torch.max(self(test_inputs), 1)
        return accuracy_score(test_labels, preds_y)

    def show_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_tracker, label="Loss Curve")
        plt.legend()
        plt.show()

    def predict(self, inputs):
        """
        Sets the model to evaluation/inference mode, disabling dropout and
        gradient calculation.
        """
        self.eval()
        return self(inputs)

    def summary(self):
        from torchsummary import summary
        summary(self, (1, 1, self.in_size))
