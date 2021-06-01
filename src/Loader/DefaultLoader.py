import numpy as np
import tensorflow as tf


def LoadDataset(datasetRoute, pretrained):
    x_train = np.load(datasetRoute + 'x_train.npy')
    x_valid = np.load(datasetRoute + 'x_valid.npy')
    x_test = np.load(datasetRoute + 'x_test.npy')
    y_train = np.load(datasetRoute + 'y_train.npy')
    y_valid = np.load(datasetRoute + 'y_valid.npy')
    y_test = np.load(datasetRoute + 'y_test.npy')

    x_train_3 = np.repeat(x_train[..., np.newaxis], 3, -1)
    x_valid_3 = np.repeat(x_valid[..., np.newaxis], 3, -1)
    x_test_3 = np.repeat(x_test[..., np.newaxis], 3, -1)

    if pretrained:
        return x_train_3, x_valid_3, x_test_3, y_train, y_valid, y_test
    else:
        return x_train, x_valid, x_test, y_train, y_valid, y_test


def LoadModel(modelRoute):
    model = tf.keras.models.load_model(modelRoute + 'trained_model.h5')

    return model
