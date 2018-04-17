import sys
import joblib
import numpy as np
import os
import pandas as pd
import numpy as np
from math import ceil

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn import preprocessing
from baseline import replace_nan
from sklearn import pipeline
from sklearn.base import TransformerMixin

pipeline_dir = "pipeline"

if not os.path.exists(pipeline_dir):
    os.mkdir(pipeline_dir)


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def final_q_indices(X_train):
    #     later find the actual indices
    from random import choices
    return choices(range(X_train.shape[0]))


def drop_inds(X_train, inds):
    return X_train.drop(inds)


def to_lstm_input(X_train, y_train, timesteps=4):
    by_sym = {}

    X_train_ = np.append(X_train, y_train, axis=1)
    for i in range(X_train.shape[0]):
        sym = X_train[i][0]
        if sym not in by_sym:
            by_sym[sym] = np.array(X_train_[i].reshape(1, 1, X_train_.shape[1]))
        else:
            by_sym[sym] = np.append(by_sym[sym], X_train_[i].reshape(1, 1, -1), axis=1)
    new_X = None
    for sym in by_sym:
        n_batches, rem = ceil(by_sym[sym].shape[1] / timesteps), int(by_sym[sym].shape[1] / timesteps)
        n_dim = by_sym[sym].shape[2]
        if rem != 0:
            for i in range(timesteps - rem):
                by_sym[sym] = np.append(by_sym[sym], np.zeros(shape=(by_sym[sym].shape[0], 1, by_sym[sym].shape[2])), axis=1)
        this_x = by_sym[sym].reshape(-1, n_batches, n_dim)
        if new_X is None:
            new_X = this_x
        else:
            for i in range(n_batches):
        new_X = np.append(new_X, this_x, axis=0) if new_X is not None else this_x
    return new_X


def model(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape), return_sequences=False)
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    print("Compiling model.")
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
    print("Done compiling model.")
    return model


def __pipeline(input_shape):
    return pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('lstm', KerasRegressor(build_fn=model(input_shape=input_shape)))
    ])


def q_num(y, q, epoch=2000):
    return ((y - epoch) * 4) + q


def sort_keys(X_train):
    to_sort = []
    for i in range(X_train.shape[0]):
        to_sort.append((i, q_num(q=X_train[i][2], y=X_train[i][1])))
    return [x[0] for x in sorted(to_sort, key=lambda k: k[1])]


def sort_batch(X_train):
    X_train_ = np.zeros(X_train.shape)
    sorted_ = sort_keys(X_train)
    for i in range(len(sorted_)):
        X_train_[i] = X_train[sorted_[i]]
    return X_train_


def sort_all_batches(X_train):
    for i in range(X_train.shape[0]):
        X_train[i] = sort_batch(X_train[i])
    return X_train


def sort_y(X_train):
    X_train_ = np.zeros(X_train.shape)
    sorted_ = sort_batch(X_train, for_q=False)
    for i in range(len(sorted_)):
        X_train_[0][i] = X_train[0][sorted_[i]]
    return X_train_


if __name__ == "__main__":
    print("Training sample neural net")
    in_file = sys.argv[1]
    print("Input file: %s" % in_file)

    data = joblib.load(in_file)
    X_, y_ = data.X_train[:-1], data.all_eps
    y_ = np.roll(y_, 1)[1:]
    replace_nan(X_)

    t = 4
    print("Original X:")
    print(X_)
    # X_ = to_lstm_input(sort_all_batches(X_), y_, timesteps=t)
    X_ = to_lstm_input(X_, y_, timesteps=t)
    print("Sorted X:")
    print(sort_all_batches(X_))


    # print("------------")
    # print("Sorting by quarter")
    # print(sort_q(first_n))
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_, y_)
    #
    # pipe = __pipeline(input_shape=X_train[0].shape)
    # pipe.fit(X_train, y_train)
    #
    # model_step = pipe.steps.pop(-1)[1]
    # scores = model_step.evaluate(X_test, y_test)
    #
    # for i in range(len(scores)):
    #     print("%s : %s" % (model_step.metrics_names[i], scores[i]))
