#!/usr/bin/env python
# coding: utf-8

import threading

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

VALIDATE_SPART = 340000
random.seed(42)
# to change according to your machine
base_dir = os.path.expanduser("data" + os.path.sep + "tum")
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
print(fid_training.keys())
print(fid_validation.keys())
print(fid_test.keys())

# get s1 image channel data
# it is not really loaded into memory. only the indexes have been loaded.
print("-" * 60)
print("training part")
s1_training = fid_training['sen1']
print(s1_training.shape)
s2_training = fid_training['sen2']
print(s2_training.shape)
label_training = fid_training['label']
print(label_training.shape)

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)
label_validation = fid_validation['label']
print(label_validation.shape)

print("-" * 60)
print("test part")
s1_test = fid_test['sen1']
print(s1_test.shape)
s2_test = fid_test['sen2']
print(s2_test.shape)

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
date_time_str = date_time.strftime('%y%m%d_%H%M')
# train data
n_sampels = s1_training.shape[0]
# validate data
val_s1_batch = np.asarray(s1_validation)
val_s2_batch = np.asarray(s2_validation)
val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
val_y = np.argmax(label_validation, axis=1)

# for concatenate
val_s1_batch_merge = s1_training[VALIDATE_SPART:, :, :, :]
val_s2_batch_merge = s2_training[VALIDATE_SPART:, :, :, :]
label_validation_merge = label_training[VALIDATE_SPART:, :]


class ThreadSafeIter():
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


@threadsafe_generator
def train_data_generator(FLAGS):
    # simple classification example
    # Training part
    n_sampels = s1_training.shape[0]
    train_y = np.argmax(label_training, axis=1)
    while True:
        # random in n_sampels
        for i in range(0, n_sampels, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_sampels)
            train_s1_tmp = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_tmp = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_X_batch = np.concatenate((train_s1_tmp, train_s2_tmp), axis=3)
            train_label = train_y[start_pos:end_pos]
            yield (train_X_batch, train_label)
            if end_pos % 2000 == 0:
                print("%s generate data %d/%d" % (datetime.now(), end_pos, n_sampels))


def train_model(model, FLAGS):
    # train start
    steps_per_epoch = n_sampels / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=4)
    checkpoint_callback = ModelCheckpoint('model' + os.path.sep + 'ckpt' + os.path.sep + '%s_%s.h5'
                                          % (FLAGS.type, date_time_str), monitor='val_acc',
                                          verbose=1, save_best_only=True)
    model.fit_generator(train_data_generator(FLAGS), epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch, verbose=2, workers=FLAGS.workers,
                        validation_data=(val_X_batch, val_y),
                        callbacks=[early_stopping_callback, checkpoint_callback])
    # model.save('model' + os.path.sep + '%s_%s.h5' %(FLAGS.type, date_time_str))
    return model


def validate_model(model, FLAGS):
    # make a prediction on validation
    # val_loss, val_acc = model.evaluate(val_X_batch, val_y)
    # print "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict(val_X_batch)
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    print(classification_report(np.argmax(label_validation, axis=1), pred_y))


def predict_result(model, FLAGS):
    # make a prediction on test
    val_s1_batch = np.asarray(s1_test)
    val_s2_batch = np.asarray(s2_test)
    val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
    pred_y = model.predict(val_X_batch)
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    # serialize
    enc = OneHotEncoder()
    enc.fit(np.arange(0, 17)[:, np.newaxis])
    ret = enc.transform(pred_y[:, np.newaxis])
    ret_df = pd.DataFrame(ret.toarray()).astype(int)
    ret_df.to_csv('result' + os.path.sep + '%s_%s.csv' % (FLAGS.type, date_time_str),
                  index=False, header=False)


# two side input
@threadsafe_generator
def train_data_generator_2(FLAGS):
    # simple classification example
    # Training part
    n_sampels = s1_training.shape[0]
    train_y = np.argmax(label_training, axis=1)
    while True:
        # random in n_sampels
        for i in range(0, n_sampels, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_sampels)
            train_s1_batch = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_label = train_y[start_pos:end_pos]
            yield ([train_s1_batch, train_s2_batch], train_label)
            if end_pos % 2000 == 0:
                print("%s generate data %d/%d" % (datetime.now(), end_pos, n_sampels))


def train_model_2(model, FLAGS):
    # train start
    steps_per_epoch = n_sampels / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=4)
    checkpoint_callback = ModelCheckpoint('model' + os.path.sep + 'ckpt' + os.path.sep + '%s_%s.h5'
                                          % (FLAGS.type, date_time_str),
                                          monitor='val_acc', verbose=1, save_best_only=True)
    model.fit_generator(train_data_generator_2(FLAGS), epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch, verbose=2, workers=FLAGS.workers,
                        validation_data=([val_s1_batch, val_s2_batch], val_y),
                        callbacks=[early_stopping_callback, checkpoint_callback])
    # model.save('model' + os.path.sep + '%s_%s.h5' %(FLAGS.type, date_time_str))
    return model


def validate_model_2(model, FLAGS):
    # make a prediction on validation
    # val_loss, val_acc = model.evaluate(val_X_batch, val_y)
    # print "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict([val_s1_batch, val_s2_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    print(classification_report(np.argmax(label_validation, axis=1), pred_y))


def predict_result_2(model, FLAGS):
    # make a prediction on test
    val_s1_batch = np.asarray(s1_test)
    val_s2_batch = np.asarray(s2_test)
    pred_y = model.predict([val_s1_batch, val_s2_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    # serialize
    enc = OneHotEncoder()
    enc.fit(np.arange(0, 17)[:, np.newaxis])
    ret = enc.transform(pred_y[:, np.newaxis])
    ret_df = pd.DataFrame(ret.toarray()).astype(int)
    ret_df.to_csv('result' + os.path.sep + '%s_%s.csv' % (FLAGS.type, date_time_str),
                  index=False, header=False)


# two side input
@threadsafe_generator
def train_data_generator_merge_2(FLAGS):
    # simple classification example
    # Training part
    n_sampels = s1_training.shape[0]
    train_y = np.argmax(label_training, axis=1)
    while True:
        # random in n_sampels
        for i in range(0, n_sampels, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_sampels)
            train_s1_batch = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_label = train_y[start_pos:end_pos]
            yield ([train_s1_batch, train_s2_batch], train_label)
            if end_pos % 2000 == 0:
                print("%s generate data %d/%d" % (datetime.now(), end_pos, n_sampels))


def train_model_merge_2(model, FLAGS):
    # train start
    steps_per_epoch = n_sampels / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=4)
    checkpoint_callback = ModelCheckpoint('model' + os.path.sep + 'ckpt' + os.path.sep + '%s_%s.h5'
                                          % (FLAGS.type, date_time_str),
                                          monitor='val_acc', verbose=1, save_best_only=True)
    model.fit_generator(train_data_generator_2(FLAGS), epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch, verbose=2, workers=FLAGS.workers,
                        validation_data=([val_s1_batch, val_s2_batch], val_y),
                        callbacks=[early_stopping_callback, checkpoint_callback])
    # model.save('model' + os.path.sep + '%s_%s.h5' %(FLAGS.type, date_time_str))
    return model


def validate_model_merge_2(model, FLAGS):
    # make a prediction on validation
    # val_loss, val_acc = model.evaluate(val_X_batch, val_y)
    # print "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict([val_s1_batch, val_s2_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    print(classification_report(np.argmax(label_validation, axis=1), pred_y))


def predict_result_merge_2(model, FLAGS):
    # make a prediction on test
    val_s1_batch = np.asarray(s1_test)
    val_s2_batch = np.asarray(s2_test)
    pred_y = model.predict([val_s1_batch, val_s2_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    # serialize
    enc = OneHotEncoder()
    enc.fit(np.arange(0, 17)[:, np.newaxis])
    ret = enc.transform(pred_y[:, np.newaxis])
    ret_df = pd.DataFrame(ret.toarray()).astype(int)
    ret_df.to_csv('result' + os.path.sep + '%s_%s.csv' % (FLAGS.type, date_time_str),
                  index=False, header=False)


def load_model(model_path):
    model = keras.models.load_model(model_path)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
