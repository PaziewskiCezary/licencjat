import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from keras.optimizers import Adam
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import random


def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 1 - 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            xw, yw = dims[0], dims[1]
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(data_dir + images[i])/255
                original_img = img_to_array(original_img)
                x = random.randint(0, original_img.shape[0]-xw)
                y = random.randint(0, original_img.shape[1]-yw)
                croped_img = np.copy(original_img[x:x+xw, y:y+yw])
                imgs.append(croped_img)

                # masks
                original_mask = load_img(mask_dir + images[i].replace('full', 'mask'))/255
                original_mask = img_to_array(original_mask)
                x = random.randint(0, original_img.shape[0]-xw)
                y = random.randint(0, original_img.shape[1]-yw)
                croped_mask = original_mask[x:x+xw, y:y+yw]
                labels.append(croped_mask)

            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)