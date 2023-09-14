from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D, LSTM
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten, Reshape, InputLayer
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import tensorflow as tf

def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))
def cnnlstm(nb_classes, Chans, Samples, dropoutRate=0.5, weight_decay=1):

    input1 = Input(shape=(Chans, Samples,1))
    print('input1: ', input1)
    print('input1 shape: ', input1.shape)
    tensor_shape =input1.shape

    # Define the fixed batch size you want to use
    fixed_batch_size = 32

    # Create the input tensor using tf.keras.Input
    input_tensor = tf.keras.Input(shape=tensor_shape[1:])
    print("test shape",input_tensor.shape)
    print(input_tensor)

    # Remove the None dimension by specifying the fixed batch size
    reshaped_tensor = tf.reshape(input_tensor,  tensor_shape[1:])
    print(reshaped_tensor)

    block1 = Conv2D(64, (1, 64), padding='same', kernel_regularizer=regularizers.l2(weight_decay),input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1) #setting the kernel length to half of the sampling rate achieves the best performance
    block1 = BatchNormalization()(block1)
    print('block1 after BN: ', block1)


    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),
                             depth_multiplier=2,
                             depthwise_constraint=max_norm(1.))(block1)
    print('block1 after depthwise: ', block1)
    block1 = BatchNormalization()(block1)
    print('block1 after BN: ', block1)
    block1 = Activation('elu')(block1)
    print('block1 after elu: ', block1)
    block1 = AveragePooling2D(pool_size=(1, 4))(block1)
    print('block1 after pooling: ', block1)
    block1 = Dropout(dropoutRate)(block1)
    print('block1 after dropout: ', block1)

    block2 = SeparableConv2D(16, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    # FOR TESTING CNN
    #flatten = Flatten(name='flatten')(block2)
    #dense = Dense(1, name='dense',kernel_constraint=max_norm(1.))(flatten)
    #sigmoid = Activation('sigmoid', name='sigmoid')(dense)
    #model = Model(inputs=input1, outputs=sigmoid)

    # Calculate the current total size
    #current_total_size = tf.reduce_prod(block1.shape)
    #print('current_total_size: ', current_total_size)
    reshape1 = Reshape((16, 16))(block2)

    lstm1 = LSTM(10, return_sequences=True)(reshape1)
    lstm1 = Dropout(dropoutRate)(lstm1)

    lstm2 = LSTM(10)(lstm1)
    lstm2 = Dropout(dropoutRate)(lstm2)
    dense = Dense(1, kernel_constraint=max_norm(0.5))(lstm2)
    print('dense: ', dense)
    #softmax = Activation('softmax')(dense)
    sigmoid = Activation('sigmoid')(dense)
    print('sigmoid: ', sigmoid)
    model = Model(inputs=input1, outputs=sigmoid)
    model.summary()
    return model
model = cnnlstm(2,19,512)
print(model.summary())