import numpy as np
import tensorflow as tf
import tensorflow.python.keras.losses as tfloss
from tensorflow.keras import layers, models
import tensorflow_addons as tfa


class Densenet121:
    def __init__(self, data, modelRoute=''):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = data
        self.modelRoute = modelRoute

    def Train(self, pre_weights, activation, learning_rate,  momentum, weight_decay, batch_size, epochs, nclasses, early_stop, save_model):

        model = tf.keras.applications.DenseNet121(weights=pre_weights, include_top=False,
                                                  input_shape=(self.x_train.shape[1], self.x_train.shape[2], 3),
                                                  pooling='avg')

        out = layers.Dense(nclasses, activation=activation)(model.output)

        full_model = models.Model(inputs=model.input, outputs=out)

        if weight_decay is None:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
        else:
            opt = tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=momentum,
                                      nesterov=True)

        if nclasses == 2:
            loss = tfloss.BinaryCrossentropy(from_logits=False)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        callbacks = self.SetCallbacks(early_stop, save_model)

        full_model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        class_weights = {0: len(self.y_train[self.y_train[:, 1] == 1]) / len(self.y_train),
                         1: len(self.y_train[self.y_train[:, 0] == 1]) / len(self.y_train)}

        history = full_model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                 callbacks=callbacks, validation_data=(self.x_valid, self.y_valid),
                                 shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
                                 steps_per_epoch=None, validation_steps=None, validation_batch_size=None,
                                 validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

        return full_model, history

    def SetCallbacks(self, early_stop, save_model):
        stopper = None
        mcp_save = None

        if early_stop:
            stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=0,
                                                       mode='auto', baseline=None, restore_best_weights=False)

        if save_model:
            mcp_save = tf.keras.callbacks.ModelCheckpoint(self.modelRoute + 'trained_model.h5',
                                                          save_best_only=True, monitor='val_loss', mode='min')

        if (stopper is not None) and (mcp_save is not None):
            return [stopper, mcp_save]
        elif stopper is not None:
            return stopper
        elif mcp_save is not None:
            return mcp_save
        else:
            return None
