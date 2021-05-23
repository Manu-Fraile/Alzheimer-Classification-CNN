import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import errno

from src.ModelAlgorithms.Densenet121 import Densenet121
from src.ModelAlgorithms.Densenet169 import Densenet169
from src.ModelAlgorithms.Densenet201 import Densenet201
from src.ModelAlgorithms.DensenetCustom import DensenetCustom


def LoadDataset(datasetRoute):
    x_train = np.load(datasetRoute + 'x_train.npy')
    x_valid = np.load(datasetRoute + 'x_valid.npy')
    x_test = np.load(datasetRoute + 'x_test.npy')
    y_train = np.load(datasetRoute + 'y_train.npy')
    y_valid = np.load(datasetRoute + 'y_valid.npy')
    y_test = np.load(datasetRoute + 'y_test.npy')

    x_train_3 = np.repeat(x_train[..., np.newaxis], 3, -1)
    x_valid_3 = np.repeat(x_valid[..., np.newaxis], 3, -1)
    x_test_3 = np.repeat(x_test[..., np.newaxis], 3, -1)

    return x_train_3, x_valid_3, x_test_3, y_train, y_valid, y_test


def SaveResults(history, model, experimentRoute):
    PlotAccuracy(history, experimentRoute)
    PlotLoss(history, experimentRoute)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    testf = open(experimentRoute + 'results.txt', 'w')
    testf.write('+++++   LAST MODEL   +++++')
    testf.write('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
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


def EvaluateModel(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('+++++   BEST MODEL   +++++')
    print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))


if __name__ == "__main__":

    # SETUP VARIABLES:
    # train
    # saveModel
    # modelRoute ex: '/models/model001/'
    # datasetRoute ex: './datasets/Data_Axial_200_Rot/'
    # datasetRoute = './datasets/Data_Axial_200_Rot/'
    # ALWAYS PUT THE LAST BAR IN ROUTES

    train = False

    modelRoute = './models/model001/'
    CheckRoute(modelRoute)
    experimentRoute = './experiments/experiment001/'
    CheckRoute(experimentRoute)
    datasetRoute = './datasets/Data_Axial_200_Rot/'

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    x_train, x_valid, x_test, y_train, y_valid, y_test = LoadDataset(datasetRoute)

    if train:
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
        pre_weights = 'imagenet'
        activation = 'softmax'
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = None
        batch_size = 32
        epochs = 40
        nclasses = 2
        early_stop = False
        save_model = True

        data = [x_train, x_valid, x_test, y_train, y_valid, y_test]
        selectedModel = Densenet121(data, modelRoute)
        #selectedModel = Densenet169(data, modelRoute)
        #selectedModel = Densenet201(data, modelRoute)
        #selectedModel = DensenetCustom(data, modelRoute)

        model, history = selectedModel.Train(pre_weights, activation, learning_rate,  momentum,
                                             weight_decay, batch_size, epochs, nclasses, early_stop,
                                             save_model)

        SaveResults(history, model, experimentRoute)

    else:
        model = LoadModel(modelRoute)

    EvaluateModel(model, x_test, y_test)



