import numpy as np
import tensorflow as tf
import tensorflow.python.keras.losses as tfloss
from tensorflow.keras import layers, models
import tensorflow_addons as tfa


class Densenet121:
    def __init__(self, data, saveModel, modelRoute=''):
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = data
        self.save = saveModel
        self.modelRoute = modelRoute

    def Train(self):
        model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False,
                                                  input_shape=(self.x_train.shape[1], self.x_train.shape[2], 3), pooling='avg')

        out = layers.Dense(2, activation='softmax')(model.output)

        full_model = models.Model(inputs=model.input, outputs=out)

        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        opt_decay = tfa.optimizers.SGDW(weight_decay=0.0008, learning_rate=0.01, momentum=0.9, nesterov=True)

        # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = tfloss.BinaryCrossentropy(from_logits=False)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=False)
        stop_at_valacc1 = tf.keras.callbacks.EarlyStopping(monitor='val_acc', baseline=1.0, patience=0)

        if self.save:
            mcp_save = tf.keras.callbacks.ModelCheckpoint('../..' + self.modelRoute + 'trained_D121.h5',
                                                          save_best_only=True,
                                                          monitor='val_loss', mode='min')
        else:
            mcp_save = None

        full_model.compile(optimizer=opt_decay, loss=loss, metrics=['accuracy'])

        class_weights = {0: len(self.y_train[self.y_train[:, 1] == 1]) / len(self.y_train),
                         1: len(self.y_train[self.y_train[:, 0] == 1]) / len(self.y_train)}

        history = full_model.fit(x=self.x_train, y=self.y_train, batch_size=32, epochs=20, verbose=1, callbacks=mcp_save,
                                 validation_data=(self.x_valid, self.y_valid), shuffle=True, class_weight=None,
                                 sample_weight=None,
                                 initial_epoch=0, steps_per_epoch=None, validation_steps=None,
                                 validation_batch_size=None,
                                 validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

        return full_model, history
