import numpy as np
from model import *
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from planar import *
import matplotlib.pyplot as plt

X, Y = sklearn.datasets.make_moons(n_samples=1500, noise=0.05)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
# train_X.shape === (nx, m)
# train_Y.shape === (1, m)
# train_X, train_Y = load_planar_dataset(400
# )
train_X = train_X.T
train_Y = train_Y.reshape(1, train_X.shape[1])
test_X = test_X.T
test_Y = test_Y.reshape(1, test_X.shape[1])
# print(train_X.shape, train_Y.shape)
neural_net = (
    Model(train_X, train_Y)
    .add_layer({"hidden_unit": 6, "activation": "Relu"})
    .add_layer({"hidden_unit": 8, "activation": "Relu"})
    .add_layer({"hidden_unit": 9, "activation": "Relu"})
    .add_layer({"hidden_unit": 1, "activation": "Sigmoid"})
    .train(initalization_method="Random", epoch=10000, learning_rate=0.3)
    .predict(test_X, test_Y)
)
