#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# to change according to your machine
base_dir = os.path.expanduser("data/tum")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_test = os.path.join(base_dir, 'round1_test_a_20181109.h5')

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
plt.imshow(10 * np.log10(s1_training[0, :, :, 4]), cmap=plt.cm.get_cmap('gray'))
plt.colorbar()
plt.title('Sentinel-1')

plt.subplot(222)
plt.imshow(s2_training[0, :, :, 1], cmap=plt.cm.get_cmap('gray'))
plt.colorbar()
plt.title('Sentinel-2')

# plt.show()

# simple classification example
# Training part
train_s1 = s1_training
train_s2 = s2_training
train_label = label_training
clf = SGDClassifier()

train_y = np.argmax(train_label, axis=1)
classes = list(set(train_y))
batch_size = 10000
n_samples = train_s1.shape[0]

for i in range(0, n_samples, batch_size):
    # this is an idea for batch training
    # you can relpace this loop for deep learning methods
    print("done %d/%d" % (i, n_samples))
    start_pos = i
    end_pos = min(i + batch_size, n_samples)
    train_s1_batch = np.asarray(train_s1[start_pos:end_pos, :, :, :])
    train_s2_batch = np.asarray(train_s2[start_pos:end_pos, :, :, :])
    cur_batch_size = train_s2_batch.shape[0]
    train_s1_batch = train_s1_batch.reshape((cur_batch_size, -1))
    train_s2_batch = train_s2_batch.reshape((cur_batch_size, -1))
    train_X_batch = np.hstack([train_s1_batch, train_s2_batch])
    # train_X_batch = train_s1_batch
    label_batch = train_y[start_pos:end_pos]
    clf.partial_fit(train_X_batch, label_batch, classes=classes)

# make a prediction on validation
pred_y = []
train_val_y = np.argmax(label_validation, axis=1)
batch_size = 10000
n_val_samples = s2_validation.shape[0]
for i in range(0, n_val_samples, batch_size):
    start_pos = i
    end_pos = min(i + batch_size, n_val_samples)
    val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
    val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
    cur_batch_size = val_s2_batch.shape[0]
    val_s1_batch = val_s1_batch.reshape((cur_batch_size, -1))
    val_s2_batch = val_s2_batch.reshape((cur_batch_size, -1))
    val_X_batch = np.hstack([val_s1_batch, val_s2_batch])
    # val_X_batch = val_s1_batch
    tmp_pred_y = clf.predict(val_X_batch)
    pred_y.append(tmp_pred_y)
pred_y = np.hstack(pred_y)

print(classification_report(train_val_y, pred_y))

# make a prediction on test
pred_y = []
batch_size = 10000
n_val_samples = s2_test.shape[0]
for i in range(0, n_val_samples, batch_size):
    start_pos = i
    end_pos = min(i + batch_size, n_val_samples)
    test_s1_batch = np.asarray(s1_test[start_pos:end_pos, :, :, :])
    test_s2_batch = np.asarray(s2_test[start_pos:end_pos, :, :, :])
    cur_batch_size = test_s2_batch.shape[0]
    test_s1_batch = test_s1_batch.reshape((cur_batch_size, -1))
    test_s2_batch = test_s2_batch.reshape((cur_batch_size, -1))
    val_X_batch = np.hstack([test_s1_batch, test_s2_batch])
    # val_X_batch = test_s1_batch
    tmp_pred_y = clf.predict(val_X_batch)
    pred_y.append(tmp_pred_y)
pred_y = np.hstack(pred_y)

# serialize
enc = OneHotEncoder()
enc.fit(np.arange(0, 17)[:, np.newaxis])
ret = enc.transform(pred_y[:, np.newaxis])
ret_df = pd.DataFrame(ret.toarray()).astype(int)
ret_df.to_csv('result' + os.path.sep + 'sgd_%s.csv' % datetime.now().strftime('%y%m%d_%H%M'), index=False, header=False)
