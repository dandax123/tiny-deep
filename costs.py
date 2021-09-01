import numpy as np


def compute_cost(AL, Y, m):

    logprobs = np.multiply(np.log(AL),Y)  + np.multiply(np.log(1-AL),1-Y)
    cost = - 1/m * np.sum(logprobs)  
    return cost


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
