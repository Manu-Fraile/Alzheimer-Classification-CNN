import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import errno

from src.ModelAlgorithms.Densenet121 import Densenet121
from src.ModelAlgorithms.Densenet169 import Densenet169
from src.ModelAlgorithms.Densenet201 import Densenet201
from src.ModelAlgorithms.DensenetCustom import DensenetCustom


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


def SaveResults(history, model, experimentRoute):
    PlotAccuracy(history, experimentRoute)
    PlotLoss(history, experimentRoute)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    testf = open(experimentRoute + 'results.txt', 'w')
    testf.write('+++++   LAST MODEL   +++++')
    testf.write('\nThe test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    testf.close()

    preds = model.predict(x_test)
    probsf = open(experimentRoute + 'probabilities.txt', 'w')
    probsf.write('The probability labels are:\n')
    probsf.write(str(preds))
    probsf.close()

    label_preds = np.argmax(preds, axis=1)
    predsf = open(experimentRoute + 'predictions.txt', 'w')
    predsf.write('The predicted labels are:\n')
    predsf.write(str(label_preds))
    predsf.close()


def PlotAccuracy(history, experimentRoute):
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')

    CheckRoute(experimentRoute + '/plots/')
    plt.savefig(experimentRoute + 'plots/accuracy.png')


def PlotLoss(history, experimentRoute):
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')

    CheckRoute(experimentRoute + '/plots')
    plt.savefig(experimentRoute + '/plots/loss.png')

def CheckRoute(filePath):
    if not os.path.exists(os.path.dirname(filePath)):
        try:
            os.makedirs(os.path.dirname(filePath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def LoadModel(file):
    model = tf.keras.models.load_model(modelRoute + 'trained_model.h5')

    return model


def EvaluateModel(model, x_test, y_test, nclasses):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('+++++   BEST MODEL   +++++')
    print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    preds = model.predict(x_test)
    label_preds = np.argmax(preds, axis=1)

    # for precision and rest
    one_hot_labels = np.zeros((len(y_test), nclasses))
    for i in range(len(y_test)):
        one_hot_labels[i, label_preds[i]] = 1

    for j in range(nclasses):
        print(f'Eval for class {j}')
        extra_metric(one_hot_labels[:, j], y_test[:, j])
        print('')


def extra_metric(predicted, true):
    TP = np.abs(np.sum(predicted * true))
    TN = np.abs(np.sum((predicted - 1) * (true - 1)))
    FP = np.abs(np.sum(predicted * (true - 1)))
    FN = np.abs(np.sum((predicted - 1) * true))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print(f'Precision: {precision}')
    print(f'Recall:    {recall}')
    print(f'f1-score:  {f1}')

    # return precision, recall, f1


if __name__ == "__main__":

    # SETUP VARIABLES:
    # train
    # model_name = 'model001'
    # experiment_name = 'experiment001'
    # dataset_name = 'Data_Axial_200_Rot'

    train = True
    model_name = 'model_ex1mod1tr1'
    experiment_name = 'ex1mod1tr1'
    dataset_name = 'Data_crop_4'

    #modelRoute = './models/' + model_name + '/'
    modelRoute = '/content/Alzheimer-Classification-CNN/models/' + model_name + '/'
    CheckRoute(modelRoute)
    #experimentRoute = './experiments/' + experiment_name + '/'
    experimentRoute = '/content/Alzheimer-Classification-CNN/experiments/' + experiment_name + '/'
    CheckRoute(experimentRoute)
    #datasetRoute = './datasets/' + dataset_name + '/'
    # datasetRoute = '/content/Alzheimer-Classification-CNN/datasets/' + dataset_name + '/'
    datasetRoute = '/content/drive/MyDrive/Colab Notebooks/' + dataset_name + '/'

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    x_train, x_valid, x_test, y_train, y_valid, y_test = LoadDataset(datasetRoute, False)
    print(x_train.shape)

    if train:
        # ------------------------------------
        # MODEL TUNING VARIABLES:
        # pre_weights = None / 'imagenet'
        # activation = 'softmax' / 'sigmoid'
        # learning_rate = float
        # momentum = float
        # weight_decay = float / None
        # batch_size = int, e.g. 32
        # epochs = int
        # classes = 2 / 4
        # early_stop = True / False
        # save_model = True / False
        # -------------------------------------
        pre_weights = 'imagenet'
        activation = 'softmax'
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 0.06
        batch_size = 64
        epochs = 40
        nclasses = 4
        gr = 32
        eps =1.001e-5
        cf=0.5
        shape=(112, 112, 1)
        dense_blocks=[6, 12, 24, 16]
        early_stop =  True
        save_model = True

        data = [x_train, x_valid, x_test, y_train, y_valid, y_test]
        #selectedModel = Densenet121(data, modelRoute)
        #selectedModel = Densenet169(data, modelRoute)
        #selectedModel = Densenet201(data, modelRoute)
        selectedModel = DensenetCustom(data, modelRoute, gr, eps, cf, shape, dense_blocks, nclasses)

        model, history = selectedModel.Train(pre_weights, activation, learning_rate,  momentum,
                                             weight_decay, batch_size, epochs, nclasses, early_stop,
                                             save_model)

        SaveResults(history, model, experimentRoute)

    else:
        model = LoadModel(modelRoute)

    # nclasses set to 2 by default
    EvaluateModel(model, x_test, y_test, 2)
