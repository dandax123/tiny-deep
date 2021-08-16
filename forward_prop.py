import numpy as np
from activations import *


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_method):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation_method == "Relu":
        A, activation_cache = relu_forward(Z)
    elif activation_method == "Sigmoid":
        A, activation_cache = sigmoid_forward(Z)
    else:
        A, activation_cache = tanh_forward(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagation(mini_batch_X, layers, parameters):
    caches = []
    A = mini_batch_X
    L = len(parameters) // 2
    for l in range(1, L + 1):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            layers[l - 1]["activation"],
        )
        caches.append(cache)
    return A, caches
