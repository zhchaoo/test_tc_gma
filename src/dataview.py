#!/usr/bin/env python
# coding: utf-8

import threading

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from datetime import datetime

TRAIN_SLICES = 7

random.seed(42)

date_time = datetime.now()
date_time_str = date_time.strftime('%y%m%d_%H%M')

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
for i in range(0, 4):
    plt.subplot(331)
    plt.imshow(s1_training[i, :, :, 0], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-0')
    plt.subplot(332)
    plt.imshow(s1_training[i, :, :, 1], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-1')
    plt.subplot(333)
    plt.imshow(np.sqrt(np.power(s1_training[i, :, :, 0], 2) + np.power(s1_training[i, :, :, 1], 2)),
               cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-01')

    plt.subplot(334)
    plt.imshow(s1_training[i, :, :, 2], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-0')
    plt.subplot(335)
    plt.imshow(s1_training[i, :, :, 3], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-1')
    plt.subplot(336)
    plt.imshow(np.sqrt(np.power(s1_training[i, :, :, 2], 2) + np.power(s1_training[i, :, :, 3], 2)),
               cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-01')

    plt.subplot(337)
    plt.imshow(s1_training[i, :, :, 6], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-0')
    plt.subplot(338)
    plt.imshow(s1_training[i, :, :, 7], cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-1')
    plt.subplot(339)
    plt.imshow(np.sqrt(np.power(s1_training[i, :, :, 6], 2) + np.power(s1_training[i, :, :, 7], 2)),
               cmap=plt.cm.get_cmap('gray'))
    plt.colorbar()
    plt.title('Sentinel-1-01')
# plt.subplot(222)
# plt.imshow(s2_training[0, :, :, 1], cmap=plt.cm.get_cmap('gray'))
# plt.colorbar()
# plt.title('Sentinel-2')
    plt.show()
exit(0)

# validate data
val_s1_batch = np.asarray(s1_validation)
val_s2_batch = np.asarray(s2_validation)
val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
val_y = np.argmax(label_validation, axis=1)
# append s1 process channel
val_s1_batch[:, :, :, 4] = np.log10(val_s1_batch[:, :, :, 4])
val_s1_batch[:, :, :, 5] = np.log10(val_s1_batch[:, :, :, 5])
val_s1_batch = np.concatenate((val_s1_batch,
                               np.expand_dims(np.sqrt(
                                   np.power(val_s1_batch[:, :, :, 0], 2) + np.power(val_s1_batch[:, :, :, 1], 2)),
                                              axis=3),
                               np.expand_dims(np.sqrt(
                                   np.power(val_s1_batch[:, :, :, 2], 2) + np.power(val_s1_batch[:, :, :, 3], 2)),
                                              axis=3),
                               np.expand_dims(np.sqrt(
                                   np.power(val_s1_batch[:, :, :, 6], 2) + np.power(val_s1_batch[:, :, :, 7], 2)),
                                              axis=3)), axis=3)

# test data
test_s1_batch = np.asarray(s1_test)
test_s2_batch = np.asarray(s2_test)
# append s1 process channel
test_s1_batch[:, :, :, 4] = np.log10(test_s1_batch[:, :, :, 4])
test_s1_batch[:, :, :, 5] = np.log10(test_s1_batch[:, :, :, 5])
test_s1_batch = np.concatenate((test_s1_batch,
                                np.expand_dims(np.sqrt(
                                    np.power(test_s1_batch[:, :, :, 0], 2) + np.power(test_s1_batch[:, :, :, 1], 2)),
                                               axis=3),
                                np.expand_dims(np.sqrt(
                                    np.power(test_s1_batch[:, :, :, 2], 2) + np.power(test_s1_batch[:, :, :, 3], 2)),
                                               axis=3),
                                np.expand_dims(np.sqrt(
                                    np.power(test_s1_batch[:, :, :, 6], 2) + np.power(test_s1_batch[:, :, :, 7], 2)),
                                               axis=3)), axis=3)

# for dataview
n_samples = s1_training.shape[0]
train_part_sep = int(n_samples / TRAIN_SLICES)
train_s1_max, train_s1_min, train_s1_mean = [], [], []
train_s2_max, train_s2_min, train_s2_mean = [], [], []
TRAIN_SHAPE_X = train_part_sep * 32 * 32
VALID_SHAPE_X = val_s1_batch.shape[0] * 32 * 32
TEST_SHAPE_X = test_s1_batch.shape[0] * 32 * 32
for i in range(0, n_samples - train_part_sep, train_part_sep):
    start_pos = i
    end_pos = i + train_part_sep
    s1_training_slice = s1_training[start_pos:end_pos, :, :, :]
    s1_training_slice[:, :, :, 4] = np.log10(s1_training_slice[:, :, :, 4])
    s1_training_slice[:, :, :, 5] = np.log10(s1_training_slice[:, :, :, 5])
    s1_training_slice = np.concatenate((s1_training_slice,
                                        np.expand_dims(np.sqrt(np.power(s1_training_slice[:, :, :, 0], 2) + np.power(
                                            s1_training_slice[:, :, :, 1], 2)), axis=3),
                                        np.expand_dims(np.sqrt(np.power(s1_training_slice[:, :, :, 2], 2) + np.power(
                                            s1_training_slice[:, :, :, 3], 2)), axis=3),
                                        np.expand_dims(np.sqrt(np.power(s1_training_slice[:, :, :, 6], 2) + np.power(
                                            s1_training_slice[:, :, :, 7], 2)), axis=3)), axis=3)
    s2_training_slice = s2_training[start_pos:end_pos, :, :, :]
    train_s1_max.append(s1_training_slice.reshape(TRAIN_SHAPE_X, 11).max(axis=0))
    train_s1_min.append(s1_training_slice.reshape(TRAIN_SHAPE_X, 11).min(axis=0))
    train_s1_mean.append(s1_training_slice.reshape(TRAIN_SHAPE_X, 11).mean(axis=0))
    train_s2_max.append(s2_training_slice.reshape(TRAIN_SHAPE_X, 10).max(axis=0))
    train_s2_min.append(s2_training_slice.reshape(TRAIN_SHAPE_X, 10).min(axis=0))
    train_s2_mean.append(s2_training_slice.reshape(TRAIN_SHAPE_X, 10).mean(axis=0))

# sen 1
print("s1_training max:", np.asarray(train_s1_max).max(axis=0))
print("s1_training min:", np.asarray(train_s1_min).min(axis=0))
print("s1_training mean:", np.asarray(train_s1_min).mean(axis=0))

print("s1_validate max:", np.asarray(val_s1_batch).reshape(VALID_SHAPE_X, 11).max(axis=0))
print("s1_validate min:", np.asarray(val_s1_batch).reshape(VALID_SHAPE_X, 11).min(axis=0))
print("s1_validate mean:", np.asarray(val_s1_batch).reshape(VALID_SHAPE_X, 11).mean(axis=0))
print("s1_validate std:", np.asarray(val_s1_batch).reshape(VALID_SHAPE_X, 11).std(axis=0))

print("s1_test max:", np.asarray(test_s1_batch).reshape(TEST_SHAPE_X, 11).max(axis=0))
print("s1_test min:", np.asarray(test_s1_batch).reshape(TEST_SHAPE_X, 11).min(axis=0))
print("s1_test mean:", np.asarray(test_s1_batch).reshape(TEST_SHAPE_X, 11).mean(axis=0))
print("s1_test std:", np.asarray(test_s1_batch).reshape(TEST_SHAPE_X, 11).std(axis=0))

# sen 2
print("s2_training max:", np.asarray(train_s2_max).max(axis=0))
print("s2_training min:", np.asarray(train_s2_min).min(axis=0))
print("s2_training mean:", np.asarray(train_s2_min).mean(axis=0))

print("s2_validate max:", np.asarray(val_s2_batch).reshape(VALID_SHAPE_X, 10).max(axis=0))
print("s2_validate min:", np.asarray(val_s2_batch).reshape(VALID_SHAPE_X, 10).min(axis=0))
print("s2_validate mean:", np.asarray(val_s2_batch).reshape(VALID_SHAPE_X, 10).mean(axis=0))
print("s2_validate std:", np.asarray(val_s2_batch).reshape(VALID_SHAPE_X, 10).std(axis=0))

print("s2_test max:", np.asarray(test_s2_batch).reshape(TEST_SHAPE_X, 10).max(axis=0))
print("s2_test min:", np.asarray(test_s2_batch).reshape(TEST_SHAPE_X, 10).min(axis=0))
print("s2_test mean:", np.asarray(test_s2_batch).reshape(TEST_SHAPE_X, 10).mean(axis=0))
print("s2_test std:", np.asarray(test_s2_batch).reshape(TEST_SHAPE_X, 10).std(axis=0))
