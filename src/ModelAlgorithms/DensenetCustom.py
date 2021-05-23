import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models


# current default init values are the ones for Densenet121
class DensenetCustom:
    def __init__(self, gr=32, eps=1.001e-5, cf=0.5, shape=(224, 224, 1), dense_blocks=None, classes=2):
        self.eps = eps
        self.cf = cf    # compression factor
        self.shape = shape
        self.classes = classes
        self.gr = gr    # growth rate defined in
                        # https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

        if dense_blocks is None:
            self.dense_blocks = [6, 12, 24, 16]
        else:
            self.dense_blocks = dense_blocks

        self.model = self.BuildModel()

    def BuildModel(self):
        # model = models.Sequential()
        inputs = keras.Input(shape=self.shape)

        # Convolution Start
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = layers.Conv2D(2 * self.gr, (7, 7), strides=2, use_bias=False)(x)  # 2*Growth rate, explained in paper
        print('C:', x.shape)

        x = layers.BatchNormalization(axis=3, epsilon=self.eps)(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        # Pooling
        x = layers.MaxPooling2D((3, 3), strides=2)(x)
        print('P:', x.shape)
        # Dense Block 1
        x = self.DenseBlock(x, self.dense_blocks[0], self.eps, self.gr)
        print('D1:', x.shape)

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
        model = keras.Model(inputs=inputs, outputs=out, name="Model 1")

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
