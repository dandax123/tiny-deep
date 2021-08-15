import math
import numpy as np
from numpy.core.records import array
from initalization import *
from forward_prop import *
from costs import *
from backward_prop import *
class Model:
    layers = []
    inputs = np.array([])
    outputs = np.array([])
    parameters = {}
    grads = {}
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs) 
        self.outputs = np.array(outputs)
    def get_layer_dims(self):
        n_x = self.inputs.shape[0]
        layers_dim = [n_x]
        for i in self.layers:
            layers_dim.append(int(i['hidden_unit']))
        return layers_dim
    def add_layer(self, layer_params):
        self.layers.append(layer_params)
        return self
    def batchify(self,mini_batch_size=64):
        mini_batches = []
        m  = self.inputs.shape[1]
        # randomize = np.arange(m)
        # np.random.shuffle(randomize)
        shuffled_X = self.inputs
        shuffled_Y = self.outputs.T
        # print(shuffled_X.shape)
        num_of_batches = math.floor(m/mini_batch_size)
        for k in range(0, num_of_batches):
            z = k*mini_batch_size
            mini_batch_X = shuffled_X[:, z: z + mini_batch_size]  
            mini_batch_Y = shuffled_Y[:, z: z + mini_batch_size]  
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            all_x = num_of_batches * mini_batch_size
            mini_batch_X =shuffled_X[:, all_x:]
            mini_batch_Y =shuffled_Y[:, all_x:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    def train(self, batch_size=50, epoch = 100000, learning_rate = 0.7, l2_decay= False, initalization_method="He"):
        mini_batches = self.batchify(batch_size)
        layer_dims = self.get_layer_dims()
        # print(layer_dims)
        self.parameters = initalize_parameter(layer_dims, initalization_method)
        for i in range(epoch):
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                AL, cache = forward_propagation(mini_batch_X, self.layers, self.parameters)
                cost = compute_cost(AL, mini_batch_Y, batch_size)
                grads = back_propagation(AL, mini_batch_Y,  cache, self.layers)
                self.parameters = update_parameters(grads, self.parameters, learning_rate)
        return self
    def predict(self):
        AL, cache = forward_propagation(self.inputs, self.layers, self.parameters)
        Y  = self.outputs.T
        predictions = 1 * [ AL > 0.5]
        predictions = np.array(predictions).reshape(1,400)
        print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
        # print(total_cost)
                






















































