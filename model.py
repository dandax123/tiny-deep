import numpy as np

class Model:
    layers = []
    inputs = np.array([])
    outputs = np.array([])
    def get_layer_dims(self):
        n_x = self.inputs.shape[0]
        layers_dim = [n_x]
        for i in self.layers:
            layers_dim.append(int(self.layers[i]['hidden_unit']))
        return layers_dim
    def add_layer(self, layer_params):
        self.layers.append(layer_params)
    def batchify(self,mini_batch_size=64):
        mini_batches = []
        m  = self.inputs.shape[1]
        randomize = np.arange(len(m))
        np.random.shuffle(randomize)
        shuffled_X = self.inputs[randomize]
        shuffled_Y = self.outputs[randomize]
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























































