import numpy as np
def compute_cost(AL, Y, m):
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)
    cost = - np.sum(logprobs) / m 
    return np.squeeze(cost)