import numpy as np
from model import *
from planar import *
import matplotlib.pyplot as plt


X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
print(X.shape, Y.shape)
neural_net = Model(X, Y.T).add_layer({'hidden_unit': 8, 'activation': "Sigmoid"}).add_layer({'hidden_unit': 6, 'activation': "Relu"}).add_layer({'hidden_unit': 4, 'activation': "Sigmoid"}).add_layer({'hidden_unit': 1, 'activation': "Sigmoid"}).train(initalization_method="He", epoch=10000).predict()