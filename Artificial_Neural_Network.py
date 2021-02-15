
from Graph_ANN import graphAnn
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.layers import GaussianNoise
from keras.utils.vis_utils import plot_model

def get_model():
    neurons =8
    model = keras.Sequential()
    model.add(layers.Dense(neurons, input_dim=4, activation="relu", name="input_layer" ,use_bias=True))
    model.add(layers.Dense(neurons, activation="relu", name="hidden_layer" ,use_bias=True))
    model.add(layers.Dense(1, activation="linear", name="output_layer", use_bias=False))
    model.compile(keras.optimizers.Adam(learning_rate=0.05), loss='mse' ,metrics=['mse' ] )
    print(model.summary())
    return model

def model_fit(model,data_train,results):  #To train the ANN with the data
    es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=50)
    history = model.fit(data_train,results,epochs=5000,verbose=0,validation_split=0.2,callbacks=[es])
    graphAnn(history.history['loss'],history.history['val_loss'])
    best_weights=model.get_weights()  #New weights that are given to the next ANN
    return best_weights

get_model()