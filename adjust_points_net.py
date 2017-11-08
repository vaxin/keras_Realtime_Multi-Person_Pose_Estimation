from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
import numpy as np
import random

def makeModel():
    model = Sequential()
    model.add(Dense(256, input_dim = 14 * 2, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(14 * 2, activation = 'relu'))
    return model

def train(model, x_train, y_train, x_val, y_val):
    batch_size = 100
    epochs = 10000
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    components = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(x_val, y_val), callbacks = [ early_stopping ])


def targetF(x):
    return x + 0.1

def rand():
    return random.randint(10, 80) / 100

def generateX(n):
    x = []
    for i in range(n):
        v = rand()
        x.append([ v ] * 28)
    return np.asarray(x)


def trainModel():
    
    x_train = generateX(1000)
    y_train = targetF(x_train)

    x_test = generateX(100)
    y_test = targetF(x_test)

    model = makeModel()
    train(model, x_train, y_train, x_test, y_test)
    model.save('model/adjust_points.h5')
    
def loadModel():
    from keras.models import load_model
    model = load_model('model/adjust_points.h5')
    return model

def predict(model, x):
    return model.predict(np.asarray(x))

if __name__ == "__main__":
    #trainModel()
    model = loadModel()
    print(predict(model, [ [ 0.6 ] * 28 ]))
