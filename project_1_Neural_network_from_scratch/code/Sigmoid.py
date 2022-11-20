#---------IMPORTS------------
import numpy as np
#----------------------------


def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    t = sigmoid_func(x)
    return t * (1-t)