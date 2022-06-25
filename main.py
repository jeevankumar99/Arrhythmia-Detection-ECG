import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, resample
from utils import *

LABELS = {
    0: 'N', 
    1: 'S',
    2: 'V', 
    3: 'F',
    4: 'Q'
}

if __name__ == "__main__":
    data_path = "mitbih_ecg_dataset"
    train_data, test_data = load_dataset(data_path)
    train_data = balance_train_data(train_data)
    