#!/usr/bin/env python
# coding: utf-8

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from sklearn.metrics import classification_report

from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

FLAGS = None
random.seed(42)
# to change according to your machine
base_dir = os.path.expanduser("data/tum")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_test = os.path.join(base_dir, 'round1_test_a_20181109.h5')
if not os.path.exists('result'):
    os.mkdir('result')
if not os.path.exists('model'):
    os.mkdir('model')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test = h5py.File(path_test, 'r')

# we can have a look at which keys are stored in the file
# you will get the return [u'label', u'sen1', u'sen2']
# sen1 and sen2 means the satellite images
print fid_training.keys()
print fid_validation.keys()
print fid_test.keys()

# get s1 image channel data
# it is not really loaded into memory. only the indexes have been loaded.
print "-" * 60
print "training part"
s1_training = fid_training['sen1']
print s1_training.shape
s2_training = fid_training['sen2']
print s2_training.shape
label_training = fid_training['label']
print label_training.shape

print "-" * 60
print "validation part"
s1_validation = fid_validation['sen1']
print s1_validation.shape
s2_validation = fid_validation['sen2']
print s2_validation.shape
label_validation = fid_validation['label']
print label_validation.shape

print "-" * 60
print "test part"
s1_test = fid_test['sen1']
print s1_test.shape
s2_test = fid_test['sen2']
print s2_test.shape

# compute the quantity for each col
label_qty = np.sum(label_training, axis=0)

# visualization, plot the first pair of Sentinel-1 and Sentinel-2 patches of training.h5
plt.subplot(212)
plt.plot(label_qty)

plt.subplot(221)
plt.imshow(np.log10(s1_training[0, :, :, 4]), cmap=plt.cm.get_cmap('gray'))
plt.colorbar()
plt.title('Sentinel-1')

plt.subplot(222)
plt.imshow(s2_training[0, :, :, 1], cmap=plt.cm.get_cmap('gray'))
plt.colorbar()
plt.title('Sentinel-2')
# plt.show()

date_time = datetime.now()


def load_model(model_path):
    model = keras.models.load_model(model_path)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model():
    # simple classification example
    # Training part
    train_s1 = []
    train_s2 = []
    train_label = []

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

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    # train_y = train_label
    batch_size = FLAGS.batch_size
    n_sampels = s1_training.shape[0]

    # random in n_sampels
    for i in range(0, n_sampels):
        # this is an idea for random training
        # you can relpace this loop for deep learning methods
        if random.random() > float(batch_size) / n_sampels:
            continue
        train_s1_tmp = np.asarray(s1_training[i, :, :, :])
        train_s2_tmp = np.asarray(s2_training[i, :, :, :])
        train_s1.append(train_s1_tmp)
        train_s2.append(train_s2_tmp)
        train_label.append(label_training[i])
    print "train data prepared!"

    train_s1 = np.array(train_s1)
    train_s2 = np.array(train_s2)
    # cur_batch_size = train_s2.shape[0]
    # train_s1_batch = train_s1.reshape((cur_batch_size, -1))
    # train_s2_batch = train_s2.reshape((cur_batch_size, -1))
    # train_X_batch = np.hstack([train_s1_batch, train_s2_batch])
    train_X_batch = np.concatenate((train_s1, train_s2), axis=3)
    label_batch = np.argmax(train_label, axis=1)
    model.fit(train_X_batch, label_batch, epochs=FLAGS.epochs)
    model.save('model/%s.h5' %date_time.strftime('%y%m%d_%H%M'))
    return model


def validate_model(model):
    # make a prediction on validation
    val_s1_batch = np.asarray(s1_validation)
    val_s2_batch = np.asarray(s2_validation)
    # cur_batch_size = val_s2_batch.shape[0]
    # val_s1_batch = val_s1_batch.reshape((cur_batch_size, -1))
    # val_s2_batch = val_s2_batch.reshape((cur_batch_size, -1))
    # val_X_batch = np.hstack([val_s1_batch, val_s2_batch])
    val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
    label_batch = np.argmax(label_validation, axis=1)
    val_loss, val_acc = model.evaluate(val_X_batch, label_batch)
    print "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict(val_X_batch)
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    print classification_report(label_batch, pred_y)


def predict_result(model):
    # make a prediction on test
    val_s1_batch = np.asarray(s1_test)
    val_s2_batch = np.asarray(s2_test)
    # cur_batch_size = val_s2_batch.shape[0]
    # val_s1_batch = val_s1_batch.reshape((cur_batch_size, -1))
    # val_s2_batch = val_s2_batch.reshape((cur_batch_size, -1))
    # val_X_batch = np.hstack([val_s1_batch, val_s2_batch])
    val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
    pred_y = model.predict(val_X_batch)
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    # serialize
    enc = OneHotEncoder()
    enc.fit(np.arange(0, 17)[:, np.newaxis])
    ret = enc.transform(pred_y[:, np.newaxis])
    ret_df = pd.DataFrame(ret.toarray()).astype(int)
    ret_df.to_csv('result/%s.csv' %date_time.strftime('%y%m%d_%H%M'),
                  index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='load last model from path')
    parser.add_argument('--batch_size', type=int, default=100000, help='train batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epoch times')
    FLAGS = parser.parse_args()

    if FLAGS.model_path:
        model = load_model(FLAGS.model_path)
    else:
        model = train_model()
    validate_model(model)
    predict_result(model)
