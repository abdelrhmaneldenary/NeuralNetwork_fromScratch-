import pandas as pd
import numpy as np

def train_validation_test(X,y,test_size,random_seed):
    np.random.seed(random_seed)
    n_samples=X.shape[0]
    indices=np.arange(n_samples)
    np.random.shuffle(indices)
    test_count=int(test_size*n_samples)
    test_indices=indices[:test_count]
    train_indices=indices[test_count:]

    X_train = X.iloc[train_indices].values
    y_train = y.iloc[train_indices].values.reshape(-1, 1)
    X_test = X.iloc[test_indices].values
    y_test = y.iloc[test_indices].values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test
