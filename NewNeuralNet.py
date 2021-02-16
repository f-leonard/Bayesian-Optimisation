from Graph_ANN import graphAnn
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import GaussianNoise
from keras.utils.vis_utils import plot_model


data_matrix = np.zeros((100,4))
i = 0
while i < len(data_matrix):
    a = 1000+(30*i)
    b = 40+(0.25*i)
    c = 2+(0.02*i)
    d = abs((a-2500)) +abs((b-52.5))+abs((c-2.000))
    d = d*-1




    data_matrix[i]= [a,b,c,d]
    i = i +1



X = data_matrix[:,:3]
y = data_matrix[:,3]
print(y[1])
print(X[1])

model = keras.Sequential()
model.add(layers.Dense(8, input_dim=3, activation="relu", name="input_layer", use_bias=True))
model.add(layers.Dense(8, activation="relu", name="hidden_layer", use_bias=True))
model.add(layers.Dense(1, activation="linear", name="output_layer", use_bias=False))
model.compile(keras.optimizers.Adam(learning_rate=0.05), loss='mse', metrics=['mse'])
model.fit(X,y,epochs = 3000, batch_size = 10)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

weights = model.get_weights()

layer1= weights[0]
np.savetxt('layer1.csv', layer1, delimiter= ',')

bias1= weights[1]
np.savetxt('bias1.csv', bias1, delimiter= ',')

layer2 = weights[2]
np.savetxt('layer2.csv', layer2, delimiter= ',')

bias2 = weights[3]
np.savetxt('bias2.csv', bias2, delimiter= ',')

layer3 = weights[4]
np.savetxt('layer3.csv', layer3, delimiter= ',')

