#Chapter 3
#from textbook Dive into Deep Learning

import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# Define Linear Regression Model
class LinearRegression(d2l.Module): 
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)  # One scalar output
        self.net.weight.data.normal_(0, 0.01)  # a mean of 0 and a sd of 0.01
        self.net.bias.data.fill_(0)  # Bias of linear layer is 0

    def forward(self, X):  # X input is passed through the linear layer self.net to generate predictions
        return self.net(X)  # Linear layer

    def loss(self, y_hat, y):  # MSE loss function
        fn = nn.MSELoss() #Subtracts predicted vs Actual
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)


model = LinearRegression(lr=0.03) 
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -4.4]), b=4.2)  # Generates fake data
trainer = d2l.Trainer(max_epochs=10)  # Creates a trainer object to handle the training
trainer.fit(model, data)

# Retrieve model parameters
w, b = model.get_w_b()
print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')

