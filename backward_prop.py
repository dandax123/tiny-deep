import numpy as np
from activations import *


def linear_backward(dZ, cache, m):
    A_prev, W, _ = cache
    dW = 1 / m * (np.dot(dZ, A_prev.T))
    db = 1 / m * (np.sum(dZ, axis=1, keepdims=True))
    dA_Prev = np.dot(W.T, dZ)
    # print(dA_Prev.shape)
    return (
        dA_Prev,
        dW,
        db,
    )


def linear_activation_backward(dA, cache, activation_method, m):
    linear_cache, activation_cache = cache
    if activation_method == "Relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation_method == "Sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        dZ = tanh_backward(activation_cache)
    return linear_backward(dZ, linear_cache, m)


def back_propagation(AL, Y, caches, layers):
    # assert(len(caches) == len(layers))
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # print(dA_prev.shape)
    # print("")
    L = len(layers)
    for l in reversed(range(1, L + 1)):
        dA = dA_prev
        (
            dA_prev,
            grads["dW" + str(l)],
            grads["db" + str(l)],
        ) = linear_activation_backward(
            dA, caches[l - 1], layers[l - 1]["activation"], m
        )
    return grads


def update_parameters(grads, params, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters
