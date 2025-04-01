import numpy as np

def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))
    
    return params

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