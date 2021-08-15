import numpy as np
def initalize_parameter(layer_dims=[], initalization_method="He"):
    L = len(layer_dims)
    parameters = {}
    for i in range(1, L):
        shape = (layer_dims[i], layer_dims[i-1])
        if(initalization_method == "He"):
            parameters['W' + str(i)]  =  initalize_he_weight(shape)
            parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def initialize_random_weight(shape):
    x = np.random.randn(shape[0], shape[1]) * 0.01
    return x

def initalize_he_weight(shape):
    x = np.random.randn(shape[0], shape[1]) * np.sqrt(2/shape[1])
    return x

