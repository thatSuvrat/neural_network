import numpy as np

def init_params(layer_len):
    params = {}
    for i in range(1, layer_len):
        params["W" + str(i)] = np.random.randn(layer_len[i],layer_len[i-1]) * 0.01
        params["b" + str(i)] = np.zeros(layer_len[i],1)
    z = np.zeros((layer_len[-1],1))
    return params, 

def activation_function(type, Z):
    if type == "sigmoid":
        return 1 / (1 + np.exp(-Z))
    elif type == "relu":
        return np.maximum(0,Z)
    elif type == "tanh":
        return np.tanh(Z)
    elif type == "softmax":
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    elif type == "softplus":
        return np.log(1 + np.exp(Z))
    else:
        raise ValueError("Unsupported activation function type: {}".format(type))

class Layer ():
    def __init__(self, layer_len, activation_type):
        self.layer_len = layer_len
        self.activation_type = activation_type
        self.params, self.z = init_params(layer_len)

    def calculateOutputs(self):
        self.z = np.dot
        self.z = activation_function(self.activation_type, self.z)

        return self.z