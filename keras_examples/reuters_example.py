import keras
from keras.datasets import reuters
from keras import models
from keras import layers

import numpy as np

print(keras.__version__)

def get_dataset():
    from keras.datasets import reuters

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def build_network():


    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax

