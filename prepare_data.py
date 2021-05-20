import numpy as np
import pandas as pd
import os

# dict for label renaming
MULTICLASS_LABEL = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
BINARY_LABEL = {0.0: 0, 0.5: 1, 1.0: 1, 2.0: 1}


def one_hot_encode(label_data, label_type='binary'):
    if label_type == 'binary':
        y = np.zeros((len(label_data), 2))
        label_data = [BINARY_LABEL[num] for num in label_data]
    else:
        y = np.zeros((len(label_data), 4))
        label_data = [MULTICLASS_LABEL[num] for num in label_data]

    for i in range(len(label_data)):
        y[i, label_data[i]] = 1
    return y


def prepare_labels(file, label_type='binary'):
    raw_label = np.load(file, allow_pickle=True)
    sorted_label = pd.DataFrame(raw_label).sort_values(by=[0])  # sort to match the mri files
    prep_label = sorted_label[1].to_numpy()  # select the labels part of the array
    one_hot_label = one_hot_encode(prep_label, label_type)
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
    folder_path = '../DataSliced_axial_full/DataSliced/'

    # get labels, they will be binary by default, if want multiclass, add 'multiclass' as argument
    y_train = prepare_labels(folder_path + 'train_labels.npy')
    y_valid = prepare_labels(folder_path + 'val_labels.npy')
    y_test = prepare_labels(folder_path + 'test_labels.npy')

    # get features aka mri slices
    x_train = prepare_mri(folder_path + 'Train/')
    x_valid = prepare_mri(folder_path + 'Val/')
    x_test = prepare_mri(folder_path + 'Test/')

    # save
    save_folder = 'axial_full_binary_data/'
    np.save(save_folder + 'y_train.npy', y_train)
    np.save(save_folder + 'y_valid.npy', y_valid)
    np.save(save_folder + 'y_test.npy', y_test)
    np.save(save_folder + 'x_train.npy', x_train)
    np.save(save_folder + 'x_valid.npy', x_valid)
    np.save(save_folder + 'x_test.npy', x_test)
