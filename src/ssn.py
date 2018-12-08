#!/usr/bin/env python
# coding: utf-8

import argparse
import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate

from utils import *


def create_model():
    # define net
    # model = keras.Sequential([
    #     keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=[32, 32, 18]),
    #     keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(128, activation=tf.nn.relu),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(17, activation=tf.nn.softmax)
    # ])

    CONCAT_AXIS = 3
    input_s1 = Input(shape=(32, 32, 8))
    input_s2 = Input(shape=(32, 32, 10))
    x1 = Conv2D(24, kernel_size=(3, 3), activation='relu')(input_s1)
    x1 = Conv2D(48, kernel_size=(3, 3), activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.2)(x1)
    x2 = Conv2D(30, kernel_size=(3, 3), activation='relu')(input_s2)
    x2 = Conv2D(60, kernel_size=(3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.2)(x2)
    x = concatenate([x1, x2], axis=CONCAT_AXIS)
    x = Flatten()(x)
    x = Dense(1323, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(17, activation='softmax')(x)

    model = Model(inputs=[input_s1, input_s2], outputs=[x])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='load last model from path')
    parser.add_argument('--batch_size', type=int, default=256, help='train batch size')
    parser.add_argument('--vt_rate', type=int, default=8, help='merge train val rate')
    parser.add_argument('--epochs', type=int, default=18, help='epoch times')
    parser.add_argument('--epochs_v', type=int, default=2, help='epoch val times')
    parser.add_argument('--steps', type=int, default=1000, help='epoch times')
    parser.add_argument('--workers', type=int, default=8, help='train workers')
    parser.add_argument('--type', type=str, default='snn', help='mode names')
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
