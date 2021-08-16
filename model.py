import math
import numpy as np
from numpy.core.records import array
from initalization import *
from forward_prop import *
from costs import *
from backward_prop import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from livelossplot import PlotLosses

liveloss = plotlosses = PlotLosses()


class Model:
    layers = []
    inputs = np.array([])
    outputs = np.array([])
    parameters = {}
    grads = {}

    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs, dtype=np.float128)
        self.outputs = np.array(outputs, dtype=np.float128)
        assert self.inputs.shape[1] == self.outputs.shape[1]
        self.parameters = {}
        self.grads = {}

    def get_layer_dims(self):
        n_x = self.inputs.shape[0]
        layers_dim = [n_x]
        for i in self.layers:
            layers_dim.append(int(i["hidden_unit"]))
        return layers_dim

    def add_layer(self, layer_params):
        self.layers.append(layer_params)
        return self

    def batchify(self, mini_batch_size=64):
        mini_batches = []
        m = self.inputs.shape[1]
        # randomize = np.arange(m)
        # np.random.shuffle(randomize)
        shuffled_X = self.inputs
        shuffled_Y = self.outputs
        # print(shuffled_X.shape)
        num_of_batches = math.floor(m / mini_batch_size)
        for k in range(0, num_of_batches):
            z = k * mini_batch_size
            mini_batch_X = shuffled_X[:, z : z + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, z : z + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            all_x = num_of_batches * mini_batch_size
            mini_batch_X = shuffled_X[:, all_x:]
            mini_batch_Y = shuffled_Y[:, all_x:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def train(
        self,
        batch_size=400,
        epoch=100000,
        learning_rate=0.5,
        l2_decay=False,
        initalization_method="He",
    ):
        mini_batches = self.batchify(batch_size)
        layer_dims = self.get_layer_dims()
        t1 = []
        t2 = []
        t3 = []
        # print(layer_dims)
        self.parameters = initalize_parameter(layer_dims, initalization_method)
        for i in range(epoch):
            logs = {}
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                # print(mini_batch_X.shape)
                AL, caches = forward_propagation(
                    mini_batch_X, self.layers, self.parameters
                )
                assert AL.shape == mini_batch_Y.shape
                cost = compute_cost(AL, mini_batch_Y, batch_size)
                grads = back_propagation(AL, mini_batch_Y, caches, self.layers)
                self.parameters = update_parameters(
                    grads, self.parameters, learning_rate
                )
                accuracy = get_accuracy_value(AL, mini_batch_Y)
                if i % 100 == 0:
                    logs = {"loss": cost, "acc": accuracy}
                    liveloss.update(logs)
                    liveloss.send()

        return self

    def predict(self, inputs, outputs):
        AL, cache = forward_propagation(inputs, self.layers, self.parameters)
        accuracy = get_accuracy_value(AL, outputs)
        print(f"Accuracy: {accuracy * 100}")
        return convert_prob_into_class(AL)
