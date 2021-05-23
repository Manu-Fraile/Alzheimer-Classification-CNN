import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models


def transition_layer(x,eps,cf):
    x = layers.BatchNormalization(axis=3, epsilon=eps)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[3] * cf), (1, 1),use_bias=False)(x)
    x = layers.MaxPooling2D((2, 2),strides = 2)(x)
    
    return(x)

def dense_block(x,n,eps,growth_rate):
    
    for i in range(n):
        xi = layers.BatchNormalization(axis=3, epsilon=eps)(x)
        xi = layers.Activation('relu')(xi)
        xi = layers.Conv2D(4*growth_rate, (1, 1),use_bias=False)(xi)
        xi = layers.BatchNormalization(axis=3, epsilon=eps)(xi)      
        xi = layers.Activation('relu')(xi)
        xi = layers.Conv2D(growth_rate, (3, 3), padding='same',use_bias=False)(xi)

        x = layers.Concatenate(axis=3)([x,xi])
    
    return(x)

def main(gr,eps,cf,shape,dense_blocks,classes):

    #model = models.Sequential()
    inputs = keras.Input(shape=shape)

    # Convolution Start
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = layers.Conv2D(2*gr, (7, 7), strides = 2,use_bias=False)(x) #2*Growth rate, explained in paper
    print('C:',x.shape)
    
    x = layers.BatchNormalization(axis=3, epsilon=eps)(x) 
    x = layers.Activation('relu')(x) 
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x) 
    # Pooling
    x = layers.MaxPooling2D((3, 3),strides=2)(x)
    print('P:',x.shape)
    # Dense Block 1 
    x = dense_block(x,dense_blocks[0],eps,gr)
    print('D1:',x.shape)
    
    # Transition Layer 1
    x = transition_layer(x,eps,cf)
    # Dense Block 2
    x = dense_block(x,dense_blocks[1],eps,gr)
    # Transition Layer 2
    x = transition_layer(x,eps,cf)
    # Dense Block 3
    x = dense_block(x,dense_blocks[2],eps,gr)
    # Transition Layer 3
    x = transition_layer(x,eps,cf)
    # Dense Block 4
    x = dense_block(x,dense_blocks[3],eps,gr)
    
    # Classification  
    x = layers.BatchNormalization(axis=3, epsilon=eps)(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(classes, activation='softmax')(x)
    
    # Building Model
    model = keras.Model(inputs=inputs, outputs=out, name="Model 1")
    
    return(model)

''' 
### EXAMPLE ###
# Variables
cf = 0.5 # compression factor, see paper
eps = 1.001e-5 # ???
gr = 32 # growth rate was defined in https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
shape = (224,224,1)
dense_blocks = [6,12,24,16]

model = main(gr,eps,cf, shape,dense_blocks,classes=2)
model.summary()
'''
