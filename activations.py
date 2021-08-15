import numpy as np

def relu_forward(z):
    return np.maximum(z, 0)

def relu_backward(z):
    return 1 * (z > 0)

def sigmoid_forward(z):
    x = 1/(1 + np.exp((-1)*z))
    return (x, z)


def tanh_forward(z):
    return (np.tanh(z),z)
def sigmoid_backward(z):
    return z

def softmax_layer(z, c):
    return z


b = sigmoid_forward(np.array([0.5, 0, 2.0]))
print(b)