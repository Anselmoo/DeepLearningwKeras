import keras
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results


def get_dataset():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000
    )
    # Our vectorized training data
    x_train = vectorize_sequences(train_data)
    #  Our vectorized test data
    x_test = vectorize_sequences(test_data)
    # Our vectorized labels
    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    return (x_train, y_train), (x_test, y_test)


def build_network():
    # Model section
    model = models.Sequential()
    model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=optimizers.RMSprop(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_validation(x_train, y_train, model):

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    acc = history.history["binary_accuracy"]
    val_acc = history.history["val_binary_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    fig, axs = plt.subplots(2, 1)
    # "bo" is for "blue dot"
    axs[0].plot(epochs, loss, "bo", label="Training loss")
    # b is for "solid blue line"
    axs[0].plot(epochs, val_loss, "b", label="Validation loss")
    axs[0].set_title("Training and validation loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, acc, "bo", label="Training acc")
    axs[1].plot(epochs, val_acc, "b", label="Validation acc")
    axs[1].set_title("Training and validation accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


def learning_evaluation(x_train, y_train, x_test, y_test):

    model = models.Sequential()
    model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    return results


if __name__ == "__main__":
    print(keras.__version__)
    train, test = get_dataset()
    model = build_network()
    train_validation(x_train=train[0], y_train=train[1], model=model)
    learning_evaluation(
        x_train=train[0], y_train=train[1], x_test=test[0], y_test=test[1]
    )
