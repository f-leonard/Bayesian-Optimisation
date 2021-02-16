import numpy as np
from numpy import genfromtxt

def environment(a,b,c):

    def relu(x):
        return np.maximum(0, x)

    layer1 = genfromtxt('layer1.csv', delimiter=',')

    bias1 = genfromtxt('bias1.csv', delimiter=',')
    bias1 = bias1.reshape(-1, 8)
    '''Layer 2 is the hidden layer'''
    layer2 = genfromtxt('layer2.csv', delimiter=',')

    bias2 = genfromtxt('bias2.csv', delimiter=',')

    bias2 = bias2.reshape(-1, 8)
    '''Layer 3 is the output layer'''
    layer3 = genfromtxt('layer3.csv', delimiter=',')
    layer3 = np.reshape(layer3, (-1, 1))


    inputs = np.array([a, b, c])
    inputs = np.reshape(inputs, (-1, 3))
    result = np.matmul(inputs, layer1)

    result = np.add(result, bias1)

    for i in range(result.size):
        result[0, i] = relu(result[0, i])
    result = np.matmul(result, layer2)
    result = np.add(result, bias2)
    for i in range(result.size):
        result[0, i] = relu(result[0, i])

    result = np.matmul(result, layer3)

    return result


def array_input(a, b, c):
    i = 0
    result_list = []

    while i < 100:
        result_list.append(float(environment(((a[i],b[i],c[i])))))
        i = i + 1
    return result_list


def environment_array(a, b, c):
    return array_input(a, b, c) if type(c) == np.ndarray else float(environment(a, b ,c))

