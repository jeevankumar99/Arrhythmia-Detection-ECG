import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, resample


def load_dataset(data_path):
    train_data = pd.read_csv(f'{data_path}/mitbih_train.csv',header=None)
    test_data = pd.read_csv(f'{data_path}/mitbih_test.csv', header=None)
    print(
        "Shape of Dataset:-\n", 
        "Train: ", train_data.shape, 
        "Test: ", test_data.shape)

    return train_data, test_data

def balance_train_data(train_data):
    labels = train_data[187].value_count()
    print("Class Distribution:-", labels)

    train_0 = train_data[train_data[187] == 0.0].sample(n=1600, random_state=1)
    train_1 = train_data[train_data[187] == 1.0]
    train_2 = train_data[train_data[187] == 2.0]
    train_3 = train_data[train_data[187] == 3.0]
    train_4 = train_data[train_data[187] == 4.0]

    train_1_bal = resample(train_1, replace=True, n_samples=1600, random_state=1)
    train_2_bal = resample(train_2, replace=True, n_samples=1600, random_state=1)
    train_3_bal = resample(train_3, replace=True, n_samples=1600, random_state=1)
    train_4_bal = resample(train_4, replace=True, n_samples=1600, random_state=1)

    balanced_train = pd.concat([train_0, train_1_bal, train_2_bal, train_3_bal, train_4_bal])
    balanced_labels = balance_train_data[187].value_count()
    print ("Class Distribution after balancing:-\n", balanced_labels)

    return balanced_train




