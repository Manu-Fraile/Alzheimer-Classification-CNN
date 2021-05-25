import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models
import tensorflow.python.keras.losses as tfloss
import tensorflow_addons as tfa


# current default init values are the ones for Densenet121
class DensenetCustom:
    def __init__(self, data, modelRoute='', gr=32, eps=1.001e-5, cf=0.5, shape=(112, 112, 1), dense_blocks=None, classes=2):
        self.eps = eps
        self.cf = cf    # compression factor
        self.shape = shape
        self.classes = classes
        print(classes)
        self.gr = gr    # growth rate defined in
                        # https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

        if dense_blocks is None:
            self.dense_blocks = [6, 12, 24, 16]
        else:
            self.dense_blocks = dense_blocks

        self.model = self.BuildModel()

        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = data
        self.modelRoute = modelRoute

    def BuildModel(self):
        # model = models.Sequential()
        inputs = keras.Input(shape=self.shape)

        # Convolution Start
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = layers.Conv2D(2 * self.gr, (7, 7), strides=2, use_bias=False)(x)  # 2*Growth rate, explained in paper

        x = layers.BatchNormalization(axis=3, epsilon=self.eps)(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        # Pooling
        x = layers.MaxPooling2D((3, 3), strides=2)(x)
        # Dense Block 1
        x = self.DenseBlock(x, self.dense_blocks[0], self.eps, self.gr)

        # Transition Layer 1
        x = self.TransitionLayer(x, self.eps, self.cf)
        # Dense Block 2
        x = self.DenseBlock(x, self.dense_blocks[1], self.eps, self.gr)
        # Transition Layer 2
        x = self.TransitionLayer(x, self.eps, self.cf)
        # Dense Block 3
        x = self.DenseBlock(x, self.dense_blocks[2], self.eps, self.gr)
        # Transition Layer 3
        x = self.TransitionLayer(x, self.eps, self.cf)
        # Dense Block 4
        x = self.DenseBlock(x, self.dense_blocks[3], self.eps, self.gr)

        # Classification
        x = layers.BatchNormalization(axis=3, epsilon=self.eps)(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.classes, activation='softmax')(x)

        # Building Model
        model = keras.Model(inputs=inputs, outputs=out)

        return model

    def TransitionLayer(self, x, eps, cf):
        x = layers.BatchNormalization(axis=3, epsilon=eps)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(int(backend.int_shape(x)[3] * cf), (1, 1), use_bias=False)(x)
        x = layers.AveragePooling2D((2, 2), strides=2)(x)

        return x

    def DenseBlock(self, x, n, eps, growth_rate):

        for i in range(n):
            xi = layers.BatchNormalization(axis=3, epsilon=eps)(x)
            xi = layers.Activation('relu')(xi)
            xi = layers.Conv2D(4*growth_rate, (1, 1), use_bias=False)(xi)
            xi = layers.BatchNormalization(axis=3, epsilon=eps)(xi)
            xi = layers.Activation('relu')(xi)
            xi = layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)(xi)

            x = layers.Concatenate(axis=3)([x, xi])

        return x

    def Train(self, pre_weights, activation, learning_rate,  momentum, weight_decay, batch_size, epochs, nclasses, early_stop, save_model):

        full_model = self.model

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

        if nclasses == 2:
            class_weights = {0: len(self.y_train[self.y_train[:, 1] == 1]) / len(self.y_train),
                             1: len(self.y_train[self.y_train[:, 0] == 1]) / len(self.y_train)}
        elif nclasses == 4:
            
            N1=len(self.y_train[self.y_train[:, 0] == 1])
            N2=len(self.y_train[self.y_train[:, 1] == 1])
            N3=len(self.y_train[self.y_train[:, 2] == 1])
            N4=len(self.y_train[self.y_train[:, 3] == 1])
            NT = 1/(1/N1+1/N2+1/N3+1/N4)
                   
            class_weights = {0: 1/N1*NT,
                             1: 1/N2*NT,
                             2: 1/N3*NT,
                             3: 1/N4*NT,}
      

        history = full_model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                 callbacks=callbacks, validation_data=(self.x_valid, self.y_valid),
                                 shuffle=True, class_weight=class_weights, sample_weight=None, initial_epoch=0,
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
