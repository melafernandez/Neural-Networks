#Code from Chapter 7.2: learning about the kernel
#Convolutions for Images
#Textbook. Dive into Deep Learning 

import torch
from torch import nn
from d2l import torch as d2l 

#the corr2d function performs a 2D convolution/2D cross-correlation
#the k (kernel) slides over the X (input) with the output of Y
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h+1, X.shape[1]-w+1))
    for i in range(Y.shape[1]):
        Y[i,j] = (X[i:i + h, j:j +w] * K).sum()
    return Y

X = torch.tensor ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X,K)

#Two parameters of a convolutional layer are the kernel and the scalar bias
class Conv2D(nn.Module):
    def __init__(self, kerne_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6,8))
X[:, 2:6] = 0 
X[:, 2:6] = 0 
X

K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X,K)
Y

#Apply the kernel to the Image 

corr2d(x.t(), K)

#Construct a two dimensional convolutional layer with 1 output channel and a kernel of shape (1,2). There is no bias

conv2d = nn.LazyConv2d(1, kernel_size=(1,2), bias = False)
#Two dimensional convolutional layer 

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))

lr = 3e-2 #learning rate

for i in range (10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)
    conv2d.zero_grad()
    l.sum().backward()
    #Update the kernel
    conv2.weight.data[:] -= lr * conv2d.weight.grad 
    if (i + 1) % 2 == 0:

        print(f'epoch {i + 1}, loss, {l.sum():.3f}')

conv2d.weight.data.reshape((1,2))








