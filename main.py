import tensorflow as tf

import src.Ensembles.DefaultEnsemble as DefaultEnsemble
import src.Evaluators.DefaultEvaluator as DefaultEvaluator
import src.DataSaving.SaveResults as SaveResults
import src.Router.RouteChecks as RouteChecks
import src.Loader.DefaultLoader as DefaultLoader
from src.ModelAlgorithms.Densenet121 import Densenet121
from src.ModelAlgorithms.Densenet169 import Densenet169
from src.ModelAlgorithms.Densenet201 import Densenet201
from src.ModelAlgorithms.DensenetCustom import DensenetCustom


if __name__ == "__main__":

    # SETUP VARIABLES:
    # train = True / False
    # ensemble_mode = True / False
    # model_name = 'model001'
    # experiment_name = 'experiment001'
    # dataset_name = 'Data_Axial_200_Rot'
    # classes = 2 / 4

    train = True
    ensemble_mode = False
    model_name = 'pruebas_manu'
    experiment_name = 'pruebas_manu'
    dataset_name = 'Data_crop_4'
    nclasses = 4
    pretrained_dataset = False

    modelRoute = './models/' + model_name + '/'
    #modelRoute = '/content/Alzheimer-Classification-CNN/models/' + model_name + '/'
    RouteChecks.CheckRoute(modelRoute)
    experimentRoute = './experiments/' + experiment_name + '/'
    #experimentRoute = '/content/Alzheimer-Classification-CNN/experiments/' + experiment_name + '/'
    RouteChecks.CheckRoute(experimentRoute)
    datasetRoute = './datasets/' + dataset_name + '/'
    # datasetRoute = '/content/Alzheimer-Classification-CNN/datasets/' + dataset_name + '/'
    #datasetRoute = '/content/drive/MyDrive/Colab Notebooks/' + dataset_name + '/'

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    x_train, x_valid, x_test, y_train, y_valid, y_test = DefaultLoader.LoadDataset(datasetRoute, pretrained_dataset)

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
        # early_stop = True / False
        # save_model = True / False
        # -------------------------------------
        pre_weights = None
        activation = 'softmax'
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 0.06
        batch_size = 64
        epochs = 2
        early_stop = True
        save_model = True

        gr = 48
        eps = 1.001e-5
        cf = 0.5
        shape = (112, 112, 1)
        dense_blocks = [6, 12, 24, 16]

        data = [x_train, x_valid, x_test, y_train, y_valid, y_test]
        #selectedModel = Densenet121(data, modelRoute)
        #selectedModel = Densenet169(data, modelRoute)
        #selectedModel = Densenet201(data, modelRoute)
        selectedModel = DensenetCustom(data, modelRoute, gr=gr, classes=nclasses)

        model, history = selectedModel.Train(pre_weights, activation, learning_rate,  momentum,
                                             weight_decay, batch_size, epochs, nclasses, early_stop,
                                             save_model)
        model = DefaultLoader.LoadModel(modelRoute)
        SaveResults.SaveResults(history, model, experimentRoute, x_test, y_test)

    elif ensemble_mode:
        modelRoute1 = './models/model_ex1mod1tr1/'
        model1 = tf.keras.models.load_model(modelRoute1 + 'trained_model.h5')
        modelRoute2 = './models/model_ex1mod2tr1/'
        model2 = tf.keras.models.load_model(modelRoute2 + 'trained_model.h5')
        modelRoute3 = './models/model_ex1mod3tr1/'
        model3 = tf.keras.models.load_model(modelRoute3 + 'trained_model.h5')

        models = [model1, model2, model3]
        vote_results = DefaultEnsemble.EnsemblePrediction(models, x_test, y_test)

    else:
        model = DefaultLoader.LoadModel(modelRoute)

    if not ensemble_mode:
        # nclasses set to 2 by default
        DefaultEvaluator.EvaluateModel(model, x_test, y_test, nclasses)
