#!/usr/bin/env python
# coding: utf-8

import argparse

from tasks import *
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input, concatenate
from keras.models import Model

# Global Constants
NB_CLASS = 17
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.4
WEIGHT_DECAY = 0.0005
LRN2D_NORM = True
DATA_FORMAT = 'channels_last'  # Theano:'channels_first' Tensorflow:'channels_last'
USE_BN = True


# normalization
def conv2D_lrn2d(x, filters, kernel_size, strides=(1, 1), padding='same', data_format=DATA_FORMAT, dilation_rate=(1, 1),
                 activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, lrn2d_norm=LRN2D_NORM, weight_decay=WEIGHT_DECAY):
    # l2 normalization
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
               dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint)(x)

    if lrn2d_norm:
        # batch normalization
        x = BatchNormalization()(x)

    return x


def inception_module(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, dilation_rate=(1, 1),
                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                     bias_constraint=None, lrn2d_norm=LRN2D_NORM, weight_decay=None):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)

    # 1x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)

    # 1x1->5x5
    # pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    # pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)

    # return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)
    return concatenate([pathway1, pathway2, pathway4], axis=concat_axis)


def create_model_inteval():
    # Data format:tensorflow,channels_last;theano,channels_last
    if DATA_FORMAT == 'channels_first':
        INP_SHAPE = (18, 32, 32)
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DATA_FORMAT == 'channels_last':
        INP_SHAPE = (32, 32, 18)
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid Dim Ordering')

    x = conv2D_lrn2d(img_input, 64, (5, 5), 1, padding='same', lrn2d_norm=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)
    x = BatchNormalization()(x)

    x = conv2D_lrn2d(x, 64, (1, 1), 1, padding='same', lrn2d_norm=False)
    x = conv2D_lrn2d(x, 192, (3, 3), 1, padding='same', lrn2d_norm=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)  # 3a
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3b
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)  # 4a
    # x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4b
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4c
    # x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4d
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 4e
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5a
    # x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5b
    x = AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', data_format=DATA_FORMAT)(x)

    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(units=NB_CLASS, activation='linear')(x)
    x = Dense(units=NB_CLASS, activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DATA_FORMAT


def create_model():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DATA_FORMAT = create_model_inteval()

    # Create a Keras Model
    model = Model(inputs=img_input, outputs=[x])
    model.summary()

    # Save a PNG of the Model Build
    # plot_model(model,to_file='GoogLeNet.png')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('Model Compiled')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='load last model from path')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epoch times')
    parser.add_argument('--steps', type=int, default=2000, help='epoch times')
    parser.add_argument('--workers', type=int, default=8, help='train workers')
    parser.add_argument('--type', type=str, default='icpt', help='mode names')
    FLAGS = parser.parse_args()

    if FLAGS.model_path:
        model = load_model(FLAGS.model_path)
    else:
        # define net
        model = create_model()
        train_model(model, FLAGS)
    validate_model(model, FLAGS)
    predict_result(model, FLAGS)
