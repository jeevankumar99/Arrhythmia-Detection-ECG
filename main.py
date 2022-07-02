import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, resample
from utils import *

from keras.utils.np_utils import to_categorical


LABELS = {
    0: 'N', 
    1: 'S',
    2: 'V', 
    3: 'F',
    4: 'Q'
}

EPOCHS = 5

if __name__ == "__main__":
    data_path = "mitbih_ecg_dataset"
    train_data, test_data = load_dataset(data_path)
    train_data = balance_train_data(train_data)

    Y_train = to_categorical(train_data[187])
    y_test = to_categorical(test_data[187])

    X_train = train_data.iloc[:, :187].values
    x_test = test_data.iloc[:, :187].values

    print ("Shape: ", x_test.shape)
    X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
    x_test = x_test.reshape(len(x_test), x_test.shape[1], 1)

    callbacks = [EarlyStopping(monitor='val_loss', patience=8), ModelCheckpoint(filepath='./best_weights.h5', monitor='val_loss', save_best_only=True)]

    model = load_model(X_train.shape)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    model.fit(
        X_train, Y_train, 
        epochs=EPOCHS, validation_data=(x_test,y_test))

    score = model.evaluate(x_test,y_test, verbose=1)
    y_predict = model.predict(x_test)

    model.save("ecg_cnn2.h5")
    print (y_predict)




    