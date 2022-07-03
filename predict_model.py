import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils.np_utils import to_categorical

if __name__ == "__main__":
    test_data = pd.read_csv('mitbih_ecg_dataset/mitbih_test.csv', header=None)

    # split test_data into labels and data
    labels = to_categorical(test_data[187])
    data = test_data.iloc[:, :187].values
    
    # Load model obtained from train
    model = tf.keras.models.load_model("ecg_cnn2.h5")

    print("\nPredicting using Saved model!\n")
    predict_score = model.evaluate(data, labels, verbose=1)
