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
import logging
from logging import handlers

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

VALIDATE_SPART = 340000
LOGGING_INTERVAL = 50
random.seed(42)
np.random.seed(42)

date_time = datetime.now()
date_time_str = date_time.strftime('%y%m%d_%H%M')
logfile_handler = handlers.RotatingFileHandler(
    'log' + os.path.sep + '%s.log' % date_time_str, maxBytes=200 * 1024 * 1024, backupCount=4)
logfile_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
console_handler = logging.StreamHandler()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logfile_handler)
logging.getLogger().addHandler(console_handler)

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
logging.info(fid_training.keys())
logging.info(fid_validation.keys())
logging.info(fid_test.keys())

# get s1 image channel data
# it is not really loaded into memory. only the indexes have been loaded.
logging.info("-" * 60)
logging.info("training part")
s1_training = fid_training['sen1']
logging.info(s1_training.shape)
s2_training = fid_training['sen2']
logging.info(s2_training.shape)
label_training = fid_training['label']
logging.info(label_training.shape)

logging.info("-" * 60)
logging.info("validation part")
s1_validation = fid_validation['sen1']
logging.info(s1_validation.shape)
s2_validation = fid_validation['sen2']
logging.info(s2_validation.shape)
label_validation = fid_validation['label']
logging.info(label_validation.shape)

logging.info("-" * 60)
logging.info("test part")
s1_test = fid_test['sen1']
logging.info(s1_test.shape)
s2_test = fid_test['sen2']
logging.info(s2_test.shape)

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

# validate data
val_s1_batch = np.asarray(s1_validation)
val_s2_batch = np.asarray(s2_validation)
val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
val_y = np.argmax(label_validation, axis=1)

# for concatenate
val_s1_merge_batch = s1_training[VALIDATE_SPART:, :, :, :]
val_s2_merge_batch = s2_training[VALIDATE_SPART:, :, :, :]
label_merge_validation = label_training[VALIDATE_SPART:, :]
val_merge_y = np.argmax(label_merge_validation, axis=1)


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


def normalize_s1(s1_input):
    s1_input[:, :, :, 4] = np.log10(s1_input[:, :, :, 4])
    s1_input[:, :, :, 5] = np.log10(s1_input[:, :, :, 5])
    return np.divide(s1_input, [120, 120, 120, 120, 4.62, 4.62, 8000, 5600])


def normalize_s2(s2_input):
    return s2_input / 2.8


def shuffle_batch(samples):
    ret_samples = []
    n_samples = samples[0].shape[0]
    index = np.arange(n_samples)
    np.random.shuffle(index)
    for sample in samples:
        if len(sample.shape) == 4:
            ret_samples.append(sample[index, :, :, :])
        elif len(sample.shape) == 2:
            ret_samples.append(sample[index, :])
        elif len(sample.shape) == 1:
            ret_samples.append(sample[index])
        else:
            raise Exception('sample shape error')
    return ret_samples


@threadsafe_generator
def train_data_generator(FLAGS):
    # simple classification example
    # Training part
    n_samples = s1_training.shape[0]
    train_y = np.argmax(label_training, axis=1)
    while True:
        # random in n_samples
        for i in range(0, n_samples, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_samples)
            train_s1_tmp = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_tmp = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_X_batch = np.concatenate((train_s1_tmp, train_s2_tmp), axis=3)
            train_label = train_y[start_pos:end_pos]
            yield (train_X_batch, train_label)
            if end_pos % (FLAGS.batch_size * LOGGING_INTERVAL) == 0:
                logging.info("%s generate data %d/%d" % (datetime.now(), end_pos, n_samples))


def train_model(model, FLAGS):
    # train start
    n_samples = s1_training.shape[0]
    steps_per_epoch = n_samples / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
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
    # logging.info "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict(val_X_batch)
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    logging.info(classification_report(np.argmax(label_validation, axis=1), pred_y))


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
    n_samples = s1_training.shape[0]
    train_y = np.argmax(label_training, axis=1)
    while True:
        # random in n_samples
        for i in range(0, n_samples, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_samples)
            train_s1_batch = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_label = train_y[start_pos:end_pos]
            yield ([train_s1_batch, train_s2_batch], train_label)
            if end_pos % (FLAGS.batch_size * LOGGING_INTERVAL) == 0:
                logging.info("%s generate data %d/%d" % (datetime.now(), end_pos, n_samples))


def train_model_2(model, FLAGS):
    # train start
    n_samples = s1_training.shape[0]
    steps_per_epoch = n_samples / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
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
    # logging.info "loss:%f accuracy:%f" % (val_loss, val_acc)

    pred_y = model.predict([val_s1_batch, val_s2_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    logging.info(classification_report(np.argmax(label_validation, axis=1), pred_y))


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
    n_samples = VALIDATE_SPART
    n_samples_v = s1_validation.shape[0]
    train_y = np.argmax(label_training, axis=1)
    j = 0
    batch_size_V = int(FLAGS.batch_size / FLAGS.vt_rate)
    batch_size_T = FLAGS.batch_size - batch_size_V
    while True:
        # random in n_samples
        for i in range(0, n_samples, batch_size_T):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + batch_size_T, n_samples)
            train_s1_batch = np.asarray(s1_training[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(s2_training[start_pos:end_pos, :, :, :])
            train_label = train_y[start_pos:end_pos]
            start_pos_v = j
            end_pos_v = min(j + batch_size_V, n_samples_v)
            val_s1_batch = np.asarray(s1_validation[start_pos_v:end_pos_v, :, :, :])
            val_s2_batch = np.asarray(s2_validation[start_pos_v:end_pos_v, :, :, :])
            val_label = val_y[start_pos_v:end_pos_v]
            train_s1_merge_batch = np.concatenate((train_s1_batch, val_s1_batch), axis=0)
            train_s2_merge_batch = np.concatenate((train_s2_batch, val_s2_batch), axis=0)
            train_merge_label = np.concatenate((train_label, val_label), axis=0)
            # update j index
            j += batch_size_V
            j = j if j < n_samples_v else 0
            if FLAGS.normalize:
                train_s1_merge_batch = normalize_s1(train_s1_merge_batch)
                train_s2_merge_batch = normalize_s2(train_s2_merge_batch)
            if FLAGS.shuffle:
                train_s1_merge_batch, train_s2_merge_batch, train_merge_label = shuffle_batch(
                    [train_s1_merge_batch, train_s2_merge_batch, train_merge_label])

            yield ([train_s1_merge_batch, train_s2_merge_batch], train_merge_label)
            if end_pos % (batch_size_T * LOGGING_INTERVAL) == 0:
                logging.info("%s generate data %d/%d" % (datetime.now(), end_pos, n_samples))


@threadsafe_generator
def train_data_generator_valid_2(FLAGS):
    # simple classification example
    # Training part
    n_samples = s1_validation.shape[0]
    train_y = np.argmax(label_validation, axis=1)
    while True:
        # random in n_samples
        for i in range(0, n_samples, FLAGS.batch_size):
            # this is an idea for random training
            # you can relpace this loop for deep learning methods
            start_pos = i
            end_pos = min(i + FLAGS.batch_size, n_samples)
            train_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
            train_label = val_y[start_pos:end_pos]
            if FLAGS.normalize:
                train_s1_batch = normalize_s1(train_s1_batch)
                train_s2_batch = normalize_s2(train_s2_batch)
            if FLAGS.shuffle:
                train_s1_batch, train_s2_batch, train_label = shuffle_batch(
                    [train_s1_batch, train_s2_batch, train_label])

            yield ([train_s1_batch, train_s2_batch], train_label)
            if end_pos % (FLAGS.batch_size * LOGGING_INTERVAL) == 0:
                logging.info("%s generate data %d/%d" % (datetime.now(), end_pos, n_samples))


def train_model_merge_2(model, FLAGS, only_valid=False):
    # train start
    n_samples = s1_training.shape[0]
    steps_per_epoch = n_samples / FLAGS.batch_size if FLAGS.steps is None else FLAGS.steps
    if only_valid:
        epochs = FLAGS.epochs_v
        generator = train_data_generator_valid_2
    else:
        epochs = FLAGS.epochs
        generator = train_data_generator_merge_2
    checkpoint_callback = ModelCheckpoint('model' + os.path.sep + 'ckpt' + os.path.sep + '%s_%s.h5'
                                          % (FLAGS.type, date_time_str),
                                          monitor='val_acc', verbose=1)
    if FLAGS.normalize:
        validate_data = ([normalize_s1(val_s1_merge_batch), normalize_s2(val_s2_merge_batch)], val_merge_y)
    else:
        validate_data = ([val_s1_merge_batch, val_s2_merge_batch], val_merge_y)

    model.fit_generator(generator(FLAGS), epochs=epochs,
                        steps_per_epoch=steps_per_epoch, verbose=2, workers=FLAGS.workers,
                        validation_data=validate_data,
                        callbacks=[checkpoint_callback])
    model.save('model' + os.path.sep + '%s_%s.h5' % (FLAGS.type, date_time_str))
    return model


def validate_model_merge_2(model, FLAGS):
    # make a prediction on validation
    # val_loss, val_acc = model.evaluate(val_X_batch, val_y)
    # logging.info "loss:%f accuracy:%f" % (val_loss, val_acc)

    if FLAGS.normalize:
        pred_y = model.predict([normalize_s1(val_s1_merge_batch), normalize_s2(val_s2_merge_batch)])
    else:
        pred_y = model.predict([val_s1_merge_batch, val_s2_merge_batch])
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = np.hstack(pred_y)
    logging.info(classification_report(np.argmax(label_merge_validation, axis=1), pred_y))


def predict_result_merge_2(model, FLAGS):
    # make a prediction on test
    if FLAGS.normalize:
        test_s1_batch = np.asarray(s1_test)
        test_s2_batch = np.asarray(s2_test)
    else:
        test_s1_batch = normalize_s1(np.asarray(s1_test))
        test_s2_batch = normalize_s2(np.asarray(s2_test))
    pred_y = model.predict([test_s1_batch, test_s2_batch])
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
