import numpy as np


def relu_forward(x):
    b = (np.maximum(0, x), x)
    return b


def relu_backward(dA, cache):

    Z = cache
    return dA * np.greater(Z, 0).astype(int)


def sigmoid_forward(x):
    s = 1 / (1 + np.exp(-x))
    c = (s, x)
    return c


def tanh_forward(z):
    d = (np.tanh(z), z)
    return d


def tanh_backward(z):
    return 1


def sigmoid_backward(dA, cache):
    Z = cache
    s, _ = sigmoid_forward(Z)
    dZ = dA * (s * (1 - s))
    return dZ


def softmax_layer(z, c):
    return z
