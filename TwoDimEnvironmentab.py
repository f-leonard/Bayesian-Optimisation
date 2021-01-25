import numpy as np


def environment(a,b):

    def relu(x):
        return np.maximum(0, x)

    #layer1 = np.array([[-3.0921824, -1.2295564, 2.524825, 0.90919493, 2.00472, 3.9612713, 1.4, 5.251755],
    #                   [3.5744455, -3.6810036, 4.8745055, 3.2572958, 6.948388, 4.12964, 2.6118754, 1.0091878],
    #                   [-1.6489224, -3.9266431, -1.5925405, -50.987362, 2.1607528, 4.4052515, -2.6973891, 1.3268412],
    #                   [-0.597243, 4.3640323, -0.8874578, 42.705992, -4.5478004, 4.8694353, 5.2138014, -3.1885703]])
    layer1 = np.array([[-2.0921824, -1.2295564, 3.524825, 0.60919493, 1.500472, 4.9612713, -0.83312845, 5.251755],
                       [3.5744455, 3.6810036, 4.8745055, 3.2572958, 4.948388, 4.12964, 1.6118754, 1.0091878],
                       [-4.6489224, -3.9266431, 3.5925405, -4.987362, 2.1607528, 4.4052515, -1.6973891, 1.3268412],
                       [-0.597243, -2.3640323, -0.8874578, 5.705992, -1.5478004, -0.8694353, -2.2138014, -1.1885703]])
    #bias1 = np.array([-2.69777, -3.854296, -1.4282128, 5.7959104, .3713766, 3.1772327, 2.1620402, 3.0930961])
    bias1 = np.array([5.69777, 4.854296, -1.4282128, 5.7959104, 0.93713766, -1.1772327, 3.1620402, 1.0930961])
    bias1 = bias1.reshape(-1, 8)

    layer2 = np.array([[1.4864122, 1.3812879, 3.3169959, 4.0733933, 3.3108673, 1.9383894, -0.8571715, 3.8677542],
                       [6.134839, 2.514021, 4.8724103, 5.6893535, 5.195739, 2.1267622, 4.8431873, 6.282418],
                       [2.3219786, 0.93685806, 1.576028, 1.0264689, 0.29711148, 0.8766245, 1.9142852, -4.828735],
                       [-0.6163249, -2.6741948, -1.9619783, -2.4506505, -1.8223758, -1.2056774, -1.1754777, 4.892866],
                       [1.7677287, 1.0984539, 3.386629, 6.0703564, 3.801893, 2.5657701, 4.231701, -0.6757402],
                       [0.64468974, 1.355038, 0.680768, 1.3085587, 0.34975052, -2.249866, -0.073497914, -3.46727],
                       [5.469417, 2.573279, 1.27536, 5.6576605, 5.33991, 5.701908, 3.3806422, 2.5719318],
                       [2.4489398, 5.095386, 4.0213246, 1.7711586, 2.5952158, 2.9393976, 3.9642975, -3.1774518]])

    bias2 = np.array([2.1337848, -0.98697954, -1.4635577, 0.7005308, 0.6077686, 1.9468734, 1.151996, 1.2201381])
    #bias2 = np.array([2.1337848, -0.98697954, -1.4635577, 0.7005308, 0.6077686, 1.9468734, 1.151996, 1.2201381])
    bias2 = bias2.reshape(-1, 8)

    layer3 = np.array([5.972767, 1.9369528, 3.6234467, 4.8388968, 4.0436954, 2.59104, 5.000915, -8.6252165])
    layer3 = np.reshape(layer3, (-1, 1))
    d = 0.1
    c = 1


    inputs = np.array([a,b,c,d])
    inputs = np.reshape(inputs, (-1, 4))
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
def abarray_input(a,b):
    i = 0
    result_list = []

    while i < 1000:
        result_list.append(float(environment(((a[i]-1000)/2500),(b[i]-45)/20)))
        i = i + 1
    return result_list
def abenvironment_array(a,b):

    return abarray_input(a,b) if type(b)== np.ndarray else float(environment(((a-1000)/2500),((b-45)/20)))
print(abenvironment_array(4000,65))