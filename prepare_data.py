import numpy as np
import pandas as pd
import os

# dict for label renaming
NUM_TO_LABEL = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


def one_hot_encode(label_data):
    y = np.zeros((len(label_data), 4))
    label_data = label_data[:, 1]
    label_data = [NUM_TO_LABEL[num] for num in label_data]
    for i in range(len(label_data)):
        y[i, label_data[i]] = 1
    return y


def prepare_labels(file):
    raw_label = np.load(file, allow_pickle=True)
    one_hot_label = one_hot_encode(raw_label)
    return one_hot_label


def prepare_mri(path):
    files = os.listdir(path)
    data_list = []
    for file in files:
        mri = pd.read_csv(path + file, header=None)
        mri = mri.to_numpy()
        data_list.append(mri)
    data = np.stack(data_list, axis=0)
    return data


if __name__ == '__main__':
    # get labels
    y_train = prepare_labels('Example_data/train_labels.npy')
    y_valid = prepare_labels('Example_data/val_labels.npy')
    y_test = prepare_labels('Example_data/test_labels.npy')

    # get features aka mri slices
    x_train = prepare_mri('Example_data/Train/')
    x_valid = prepare_mri('Example_data/Val/')
    x_test = prepare_mri('Example_data/Test/')

    # save
    save_folder = 'Example_data_preprocessed/'
    np.save(save_folder + 'y_train.npy', y_train)
    np.save(save_folder + 'y_valid.npy', y_valid)
    np.save(save_folder + 'y_test.npy', y_test)
    np.save(save_folder + 'x_train.npy', x_train)
    np.save(save_folder + 'x_valid.npy', x_valid)
    np.save(save_folder + 'x_test.npy', x_test)
