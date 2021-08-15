import numpy as np
from activations import *

def linear_backward(dZ, cache, m):
    # print(A_prev.T.shape)
    A_prev, W, _ = cache
    dW = 1/m * (np.dot(dZ, A_prev.T))
    db = 1/m * (np.sum(dZ, axis = 1, keepdims=True))
    dA_Prev = (np.dot(W.T, dZ))
    return dA_Prev, dW, db,

def linear_activation_backward(dA, cache, activation_method, m):
    linear_cache, activation_cache = cache
    if(activation_method == "Relu"):
        dZ = relu_backward(dA, activation_cache)
    elif(activation_method == "Sigmoid"):
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        dZ =  dA * tanh_backward(activation_cache) 
    # print(dZ.shape)
    dA_Prev, dW, db = linear_backward(dZ, linear_cache, m)
    return dA_Prev, dW, db

def back_propagation(AL, Y, caches, layers):
    # assert(len(caches) == len(layers))
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    # print(L?>?<<>?M?)
    # Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # grads['dA' + str(L)] = dAL
    dA_prev, dW, db = linear_activation_backward(dAL, caches[L-1], layers[L-1]['activation'],m)
    # print("")
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    for l in reversed(range(L-1)):
        dA_prev, dW, db = linear_activation_backward(grads['dA' + str(l+1)], caches[l], layers[l]['activation'],m)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db
    return grads

def update_parameters(grads, params, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2 
    for l in range(1, L + 1):
        # print(grads["dW" + str(l)].shape)
        parameters["W" + str(l)] -=  learning_rate * grads['dW' + str(l)]
        parameters["b" + str(l)] -=  learning_rate * grads['db' + str(l)]
    return parameters