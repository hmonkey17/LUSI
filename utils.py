import numpy as np
from sklearn.utils.multiclass import unique_labels

def load_data(filename, train_size_per_class = 200):
    file = np.load("mnist.npz")
    X_train = file['x_train']
    y_train = file['y_train']
    X_test = file['x_test']
    y_test = file['y_test']
    
    X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype(np.float64) / 255
    X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype(np.float64) / 255
    
    n_samples = X_train.shape[0]
    classes_ = unique_labels(y_train)
    n_classes = len(classes_)

    np.random.seed(2222)
    perm = np.random.permutation(n_samples)
    X = X_train[perm]
    y = y_train[perm]

    train_count = np.zeros(n_classes)
    train_idx = []

    for i in range(n_samples):
        label_index = y[i]
        if train_count[label_index] < train_size_per_class:
            train_idx.append(i)
            train_count[label_index] += 1

        if train_count.min() == train_size_per_class:
            break
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    
    return X_train, y_train, X_test, y_test