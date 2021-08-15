import numpy as np

def relu_forward(z):
    b = (np.maximum(z, 0),z )
    return b
def relu_backward(dA, cache):
     
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_forward(x):
    s = 1/(1+np.exp(-x)) 
    c =  (s, x)
    return c

def tanh_forward(z):
    d = (np.tanh(z),z)
    return d
def tanh_backward(z):
    return 1
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def softmax_layer(z, c):
    return z

