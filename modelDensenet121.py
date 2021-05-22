import numpy as np
import tensorflow as tf
import tensorflow.python.keras.losses as tfloss
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


def LoadDataset():
    x_train = np.load('./Datasets/Data_Axial_200_Rot/x_train.npy')
    x_valid = np.load('./Datasets/Data_Axial_200_Rot/x_valid.npy')
    x_test = np.load('./Datasets/Data_Axial_200_Rot/x_test.npy')
    y_train = np.load('./Datasets/Data_Axial_200_Rot/y_train.npy')
    y_valid = np.load('./Datasets/Data_Axial_200_Rot/y_valid.npy')
    y_test = np.load('./Datasets/Data_Axial_200_Rot/y_test.npy')

    x_train_3 = np.repeat(x_train[..., np.newaxis], 3, -1)
    x_valid_3 = np.repeat(x_valid[..., np.newaxis], 3, -1)
    x_test_3 = np.repeat(x_test[..., np.newaxis], 3, -1)

    return x_train_3, x_valid_3, x_test_3, y_train, y_valid, y_test


def SaveResults(history, model):
    PlotAccuracy(history)
    PlotLoss(history)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    testf = open('./Results_Densenet121/weight_decay_testResults121.txt', 'w')
    testf.write('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    testf.close()

    preds = model.predict(x_test)
    probsf = open('./Results_Densenet121/weight_decay_probabilities121.txt', 'w')
    probsf.write('The probability labels are:\n')
    probsf.write(str(preds))
    probsf.close()

    label_preds = np.argmax(preds, axis=1)
    predsf = open('./Results_Densenet121/weight_decay_predictions121.txt', 'w')
    predsf.write('The predicted labels are:\n')
    predsf.write(str(label_preds))
    predsf.close()


def PlotAccuracy(history):
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    plt.savefig('./Results_Densenet121/plots/weight_decay_accuracy121.png')


def PlotLoss(history):
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    plt.savefig('./Results_Densenet121/plots/weight_decay_loss121.png')


if __name__ == '__main__':

    retrain = False

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    x_train, x_valid, x_test, y_train, y_valid, y_test = LoadDataset()

    if retrain:

        model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False,
                                                  input_shape=(x_train.shape[1], x_train.shape[2], 3), pooling='avg')

        out = layers.Dense(2, activation='softmax')(model.output)

        full_model = models.Model(inputs=model.input, outputs=out)

        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        opt_decay = tfa.optimizers.SGDW(weight_decay=0.0008, learning_rate=0.01, momentum=0.9, nesterov=True)

        #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = tfloss.BinaryCrossentropy(from_logits=False)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=False)
        stop_at_valacc1 = tf.keras.callbacks.EarlyStopping(monitor='val_acc', baseline=1.0, patience=0)
        mcp_save = tf.keras.callbacks.ModelCheckpoint('./Models/Densenet121/weight_decay_best.h5', save_best_only=True, monitor='val_loss', mode='min')

        full_model.compile(optimizer=opt_decay, loss=loss, metrics=['accuracy'])


        class_weights = {0: len(y_train[y_train[:, 1] == 1]) / len(y_train),
                         1: len(y_train[y_train[:, 0] == 1]) / len(y_train)}

        #print('\n\nTHE WEIGHTS OF THE CLASSES ARE:' )
        #print(class_weights)

        history = full_model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, verbose=1, callbacks=mcp_save,
                                 validation_data=(x_valid, y_valid), shuffle=True, class_weight=None, sample_weight=None,
                                 initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None,
                                 validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

        #best_model = save_best_model.best_model

        SaveResults(history, full_model)

        #full_model.save('./Models/Densenet121/weight_decay.h5')

    else:
        model = tf.keras.models.load_model('./Models/Densenet121/weight_decay_best.h5')
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))

