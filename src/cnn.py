#!/usr/bin/env python
# coding: utf-8

import argparse
import tensorflow as tf

from tasks import *


def create_model():
    # define net
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=[32, 32, 18]),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(17, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='load last model from path')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epoch times')
    parser.add_argument('--steps', type=int, default=2000, help='epoch times')
    parser.add_argument('--workers', type=int, default=4, help='train workers')
    parser.add_argument('--type', type=str, default='cnn', help='mode names')
    FLAGS = parser.parse_args()

    if FLAGS.model_path:
        model = load_model(FLAGS.model_path)
    else:
        # define net
        model = create_model()
        train_model(model, FLAGS)
    validate_model(model, FLAGS)
    predict_result(model, FLAGS)
