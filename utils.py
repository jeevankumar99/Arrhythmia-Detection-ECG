import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, resample

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint


def load_dataset(data_path):
    train_data = pd.read_csv(f'{data_path}/mitbih_train.csv',header=None)
    test_data = pd.read_csv(f'{data_path}/mitbih_test.csv', header=None)
    print(
        "Shape of Dataset:-\n", 
        "Train: ", train_data.shape, 
        "Test: ", test_data.shape)

    return train_data, test_data

def balance_train_data(train_data):
    labels = train_data[187].value_counts()
    print("Class Distribution:-\n", labels)

    train_0 = train_data[train_data[187] == 0.0].sample(n=16000, random_state=1)
    train_1 = train_data[train_data[187] == 1.0]
    train_2 = train_data[train_data[187] == 2.0]
    train_3 = train_data[train_data[187] == 3.0]
    train_4 = train_data[train_data[187] == 4.0]

    train_1_bal = resample(train_1, replace=True, n_samples=16000, random_state=1)
    train_2_bal = resample(train_2, replace=True, n_samples=16000, random_state=1)
    train_3_bal = resample(train_3, replace=True, n_samples=16000, random_state=1)
    train_4_bal = resample(train_4, replace=True, n_samples=16000, random_state=1)

    balanced_train = pd.concat([train_0, train_1_bal, train_2_bal, train_3_bal, train_4_bal])
    balanced_labels = balanced_train[187].value_counts()

    n_sample = train_0.sample(600)
    s_sample = train_1.sample(600)
    v_sample = train_2.sample(600)
    f_sample = train_3.sample(600)
    q_sample = train_4.sample(600)

    fig, axis = plt.subplots(5, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)

    axis[0].set_title("None echoic beats")
    axis[0].plot(n_sample.iloc[0, :186])

    axis[1].set_title("Superventricular Echoic beats")
    axis[1].plot(s_sample.iloc[0, :186])

    axis[2].set_title("Ventricular Ectopic beats")
    axis[2].plot(v_sample.iloc[0, :186])

    axis[3].set_title("Fusion beats")
    axis[3].plot(f_sample.iloc[0, :186])

    axis[4].set_title("Unknown beats")
    axis[4].plot(q_sample.iloc[0, :186])

    plt.show()

    print ("Class Distribution after balancing:-\n", balanced_labels)

    return balanced_train


def load_model(train_shape):
    model = Sequential()
    model.add(Conv1D(128,3,input_shape=(train_shape[1],1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(64,3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(64,2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(64,2, activation='relu'))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    return model


