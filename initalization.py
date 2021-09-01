import numpy as np


def initalize_parameter(layer_dims=[], initalization_method="He"):
    L = len(layer_dims)
    parameters = {}

    for i in range(1, L):
        shape = (layer_dims[i], layer_dims[i - 1])
        if initalization_method == "He":
            parameters["W" + str(i)] = np.random.uniform(
                -1.1, 1.0, size=(layer_dims[i], layer_dims[i - 1])
            ) * np.sqrt(2 / layer_dims[i - 1]).astype(np.float16)
            parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
        else:
            parameters["W" + str(i)] = (
                np.random.randn(shape[0], shape[1]).astype(np.float16) * 0.01
            )
            parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
    # for i in range(1, L):
    #     x, y = parameters["W" + str(i)].shape
    #     print(f"W {str(i)} ({x}, {y})")
    return parameters
