import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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


def SaveResults(history, model, modelRoute):
    PlotAccuracy(history, modelRoute)
    PlotLoss(history, modelRoute)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    testf = open('./experiments/experiment001/results.txt', 'w')
    testf.write('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    testf.close()

    preds = model.predict(x_test)
    probsf = open('./experiments/experiment001/probabilities.txt', 'w')
    probsf.write('The probability labels are:\n')
    probsf.write(str(preds))
    probsf.close()

    label_preds = np.argmax(preds, axis=1)
    predsf = open('./experiments/experiment001/predictions121.txt', 'w')
    predsf.write('The predicted labels are:\n')
    predsf.write(str(label_preds))
    predsf.close()


def PlotAccuracy(history, modelRoute):
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    plt.savefig('.' + modelRoute + 'accuracy121.png')


def PlotLoss(history, modelRoute):
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    plt.savefig('.' + modelRoute + 'loss121.png')


def LoadModel(file):
    model = tf.keras.models.load_model(file)

    return model


def EvaluateModel(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))


if __name__ == "__main__":

    # Variables to manually custom:
    # train
    # saveModel
    # modelRoute ex: '/models/model001/'
    # datasetRoute ex: './datasets/Data_Axial_200_Rot/'
    # ALWAYS PUT THE LAST BAR IN ROUTES

    train = True
    modelRoute = '/models/model001/'
    datasetRoute = './datasets/Data_Axial_200_Rot/'

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    x_train, x_valid, x_test, y_train, y_valid, y_test = LoadDataset(datasetRoute)

    if train:
        saveModel = False

        data = [x_train, x_valid, x_test, y_train, y_valid, y_test]
        selectedModel = Densenet121(data, saveModel)

        model, history = selectedModel.Train()

        PlotAccuracy(history, modelRoute)
        PlotAccuracy(history, modelRoute)

    else:
        model = LoadModel(modelRoute)

    EvaluateModel(model, x_test, y_test)



