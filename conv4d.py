from keras.layers import (
    Input,
    Activation,
    Lambda,
    add,
    Concatenate,
GlobalAveragePooling3D,
Dense,
Reshape

)
import keras
from keras.layers.convolutional import Convolution3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


K.set_image_data_format('channels_first')





def expand0(input):
    x = input[:,:,:,:,:,0]
    return x
def expand1(input):
    x = input[:,:,:,:,:,1]
    return x
def expand2(input):
    x = input[:,:,:,:,:,2]
    return x
def jh(input):
    x = tf.transpose(input,(0,2,1,3,4))
    return  x


import tensorflow as tf
def SeNet(nb_filter):
    def f(input):
        squeeze0 = GlobalAveragePooling3D()(input[:, :, :, :, :, 0])  # none,64
        squeeze1 = GlobalAveragePooling3D()(input[:, :, :, :, :, 1])
        squeeze2 = GlobalAveragePooling3D()(input[:, :, :, :, :, 2])
        squeeze = (squeeze0 + squeeze1 + squeeze2) / 3
        print(squeeze.shape)
        excitation = Dense(nb_filter)(squeeze)  # none,16
        print(excitation.shape)
        excitation =  Activation('tanh')(excitation)  # none，16
        excitation = Dense(input.shape[-5])(excitation)  # none,64
        excitation = Activation('sigmoid')(excitation)  # none,64
        excitation = Reshape((input.shape[-5], 1, 1, 1, 1))(excitation)  # none，64,1，1,1
        scale = input * excitation
        return scale
    return f



def conv4d(c_conf=(2,6,25,20,3)):
    nb_flow, len_closeness,map_height, map_width,d = c_conf
    main_inputs = []

    input = Input(shape=(nb_flow, len_closeness, map_height, map_width,d))  # (2,t_c,h,# w)
    main_inputs.append(input)


    shuru0 = Lambda(expand0)(input)
    shuru1 = Lambda(expand1)(input)
    shuru2 = Lambda(expand2)(input)


    mout0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru0)
    mout1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru1)
    mout2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru2)
    m_out0 = add([mout0, mout1,mout2])
    m_out0 = Activation("tanh")(m_out0)
    mout0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru0)
    mout1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru1)
    mout2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru2)
    m_out1 = add([mout0,mout1,mout2])
    m_out1 = Activation("tanh")(m_out1)
    mout0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru0)
    mout1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru1)
    mout2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(shuru2)
    m_out2 = add([mout0,mout1, mout2])
    m_out2 = Activation("tanh")(m_out2)
    #
    m0 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_out0)
    m1 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_out1)
    m2 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_out2)
    c0 = Concatenate(axis=5)([m0, m1, m2])
    print('c0',c0.shape)#none,64,6,25,20,3
    C = Lambda(SeNet(nb_filter=16))(c0)
    m_out0 = Lambda(expand0)(C)
    m_out1 = Lambda(expand1)(C)
    m_out2 = Lambda(expand2)(C)

    m0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out0)
    m1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out1)
    m2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out2)
    m_0 = add([m0, m1, m2])
    m_0 = Activation("tanh")(m_0)
    m0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out0)
    m1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out1)
    m2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out2)
    m_1 = add([m0, m1, m2])
    m_1 = Activation("tanh")(m_1)
    m0 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out0)
    m1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out1)
    m2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(m_out2)
    m_2 = add([m0, m1, m2])
    m_2 = Activation("tanh")(m_2)

    m0 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_0)
    m1 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_1)
    m2 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_2)
    c0 = Concatenate(axis=5)([m0, m1, m2])
    C = Lambda(SeNet(nb_filter=16))(c0)
    m_0 = Lambda(expand0)(C)
    m_1 = Lambda(expand1)(C)
    m_2 = Lambda(expand2)(C)


    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_00 = add([mout0, mout1,mout2])
    m_00 = Activation("linear")(m_00)
    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_11 = add([mout0, mout1, mout2])
    m_11 = Activation("linear")(m_11)
    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_22 = add([mout0,mout1, mout2])
    m_22 = Activation("linear")(m_22)

    m_00 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_00)
    m_11 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_11)
    m_22 = Lambda(lambda x: x[:,:,:,:,:, tf.newaxis])(m_22)
    output_c0 = Concatenate(axis=5)([m_00, m_11,m_22])

    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_00 = add([mout0, mout1,mout2])
    m_00 = Activation("linear")(m_00)
    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_11 = add([mout0, mout1, mout2])
    m_11 = Activation("linear")(m_11)
    mout0 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_0)
    mout1 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_1)
    mout2 = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode="same")(m_2)
    m_22 = add([mout0,mout1, mout2])
    m_22 = Activation("linear")(m_22)

    m_00 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_00)
    m_11 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_11)
    m_22 = Lambda(lambda x: x[:, :, :, :, :, tf.newaxis])(m_22)
    output_c1 = Concatenate(axis=5)([m_00, m_11, m_22])
    output_c = Concatenate(axis=1)([output_c0,output_c1])


    print(output_c)
    print(input)
    model = Model(inputs = input, outputs = output_c)

    return model
model = conv4d()






