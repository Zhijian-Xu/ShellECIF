import pandas as pd
import numpy as np
import os
import pickle

x_train_path = './features/x_train.pkl'
y_train_path = './features/y_train.pkl'
x_test_path = './features/x_test.pkl'
y_test_path = './features/y_test.pkl'

USE_SAVED_FEATURES = True
SHELLS = 23
FEATURE_LEN = 1540

def get_shell_features():
    if USE_SAVED_FEATURES: return load_saved_features()
    print('please run generate_features.py first')
    exit(0)


def handle_shells(X):
    # generate various shells features
    X = np.array(X).reshape((-1, SHELLS, FEATURE_LEN))
    for index in range(X.shape[0]):
        for shell in range(SHELLS-1, 0, -1):
            if (X[index][shell] >= X[index][shell-1]).all():
                X[index][shell] -= X[index][shell-1]
            else:
                raise Exception(f"shell {shell} is smaller than shell {shell-1}")
    # reshape -> flatten for machine learning based on desicion tree
    return X.reshape(-1, SHELLS * FEATURE_LEN)


def save_features():
    x_train, y_train, x_test, y_test = get_shell_features()
    y_train, y_test = np.array(y_train), np.array(y_test)
    with open(x_train_path, 'wb') as f:
        pickle.dump(x_train, f)
    with open(x_test_path, 'wb') as f:
        pickle.dump(x_test, f)
    with open(y_train_path, 'wb') as f:
        pickle.dump(y_train, f)
    with open(y_test_path, 'wb') as f:
        pickle.dump(y_test, f)

def load_saved_features():
    with open(x_train_path, 'rb') as f:
        x_train = pickle.load(f)
    with open(x_test_path, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)
    return x_train, y_train, x_test, y_test
