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
CONCAT_AXIS = 3
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
    (branch1, branch2, branch4) = params
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

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1, pathway2, pathway4], axis=concat_axis)


def inception_net(img_input, base_kn = 10.67):
    # Data format:tensorflow,channels_last;theano,channels_last
    x = conv2D_lrn2d(img_input, int(base_kn * 6), (5, 5), 1, padding='same', lrn2d_norm=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)
    x = BatchNormalization()(x)

    x = conv2D_lrn2d(x, int(base_kn * 6), (1, 1), 1, padding='same', lrn2d_norm=False)
    x = conv2D_lrn2d(x, int(base_kn * 18), (3, 3), 1, padding='same', lrn2d_norm=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(int(base_kn * 6),), (int(base_kn * 9), int(base_kn * 12)), (int(base_kn * 3),)], concat_axis=CONCAT_AXIS)  # 3a
    x = inception_module(x, params=[(int(base_kn * 12),), (int(base_kn * 12), int(base_kn * 18)), (int(base_kn * 6),)], concat_axis=CONCAT_AXIS)  # 3b
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(int(base_kn * 18),), (int(base_kn * 9), int(base_kn * 19.5)), (int(base_kn * 6),)], concat_axis=CONCAT_AXIS)  # 4a
    x = inception_module(x, params=[(int(base_kn * 12),), (int(base_kn * 12), int(base_kn * 24)), (int(base_kn * 6),)], concat_axis=CONCAT_AXIS)  # 4c
    x = inception_module(x, params=[(int(base_kn * 24),), (int(base_kn * 16), int(base_kn * 30)), (int(base_kn * 12),)], concat_axis=CONCAT_AXIS)  # 4e
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(int(base_kn * 24),), (int(base_kn * 16), int(base_kn * 30)), (int(base_kn * 12),)], concat_axis=CONCAT_AXIS)  # 5a
    x = AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', data_format=DATA_FORMAT)(x)

    return x


def create_model():
    # Create the Model
    s1_input = Input(shape=(32, 32, 8))
    s2_input = Input(shape=(32, 32, 10))
    s1 = inception_net(s1_input, base_kn=8)
    s2 = inception_net(s2_input, base_kn=10)
    x = concatenate([s1, s2], axis=CONCAT_AXIS)
    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(units=NB_CLASS, activation='linear')(x)
    x = Dense(units=NB_CLASS, activation='softmax')(x)

    # Create a Keras Model
    model = Model(inputs=[s1_input, s2_input], outputs=[x])
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
    parser.add_argument('--batch_size', type=int, default=256, help='train batch size')
    parser.add_argument('--vt_rate', type=int, default=8, help='merge train val rate')
    parser.add_argument('--epochs', type=int, default=20, help='epoch times')
    parser.add_argument('--epochs_v', type=int, default=20, help='epoch val times')
    parser.add_argument('--steps', type=int, default=1000, help='epoch times')
    parser.add_argument('--steps_v', type=int, default=100, help='epoch times')
    parser.add_argument('--workers', type=int, default=8, help='train workers')
    parser.add_argument('--type', type=str, default='snn', help='mode names')
    parser.add_argument('--normalize', default=False, action='store_true', help='enable normalize')
    parser.add_argument('--shuffle', default=False, action='store_true', help='enable normalize')
    parser.add_argument('--train_spart', type=int, default=340000)
    # parser.add_argument('--valid_spart', type=int, default=22000)
    parser.add_argument('--valid_spart', type=int, default=valid_default_spart)
    parser.add_argument('--early_stop', type=int, default=-1)
    FLAGS = parser.parse_args()

    if FLAGS.model_path:
        model = load_model(FLAGS.model_path)
    else:
        # define net
        model = create_model()
        train_model_merge_2(model, FLAGS, False)
        validate_model_merge_2(model, FLAGS)
        train_model_merge_2(model, FLAGS, True)
    validate_model_merge_2(model, FLAGS)
    predict_result_merge_2(model, FLAGS)

