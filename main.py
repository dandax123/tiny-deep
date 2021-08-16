import numpy as np
from model import *
import sklearn
import sklearn.datasets
from planar import *
import matplotlib.pyplot as plt

train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)
neural_net = (
    Model(train_X.T, train_Y.reshape(train_X.shape[0], 1))
    .add_layer({"hidden_unit": 4, "activation": "Relu"})
    .add_layer({"hidden_unit": 1, "activation": "Sigmoid"})
    .train(initalization_method="He", epoch=1000)
    .predict(test_X.T, test_Y)
)
