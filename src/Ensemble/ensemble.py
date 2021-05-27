import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import mode
from tensorflow.keras.models import load_model


# models is a list of model objects
def ensemble_prediction(models, x_test, y_test):
    # collect predictions from all models
    labels = []
    for model in models:
        preds = model.predict(x_test)
        label_preds = np.argmax(preds, axis=1)
        labels.append(label_preds)
    # find the majority vote
    all_labels = np.stack(labels, axis=1)
    majority_vote = mode(all_labels, axis=1)[0].reshape(len(all_labels))
    acc = np.sum(y_test[:, 1] == majority_vote) / len(y_test)
    print(f'Ensemble accuracy: {acc}')
    return majority_vote


modelRoute1 = '../../models/model_ex1mod1tr1/'
model1 = tf.keras.models.load_model(modelRoute1 + 'trained_model.h5')
print(modelRoute1 + 'trained_model.h5')
modelRoute2 = '../../models/model_ex1mod2tr1/'
model2 = tf.keras.models.load_model(modelRoute2 + 'trained_model.h5')
print(modelRoute2 + 'trained_model.h5')

modelRoute3 = '../../models/model_ex1mod3tr1/'
model3 = tf.keras.models.load_model(modelRoute3 + 'trained_model.h5')
print(modelRoute3 + 'trained_model.h5')


models = [model1, model2, model3]

dataset_name = 'Data_crop_4'
datasetRoute = '../../' + dataset_name + '/'
x_test = np.load(datasetRoute + 'x_test.npy')
y_test = np.load(datasetRoute + 'y_test.npy')

ensemble_prediction(models, x_test, y_test)
