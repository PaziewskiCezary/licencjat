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
                original_img = load_img(data_dir + images[i])
                resized_img = imresize(original_img, dims+[3])
                array_img = img_to_array(resized_img)/255

                imgs.append(array_img)

                # masks

                original_mask = load_img(mask_dir + images[i].replace('full', 'mask'))
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_img)/255
                labels.append(array_mask[:, :, 0])

            imgs = np.array(imgs)
            labels = np.array(labels)
#             labels = labels.reshape(-1, dims[0], dims[1], 1)

            yield imgs, labels
            

def data_gen_small_croped(data_dir, mask_dir, images, batch_size, dims):
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            xw, yw = dims[0], dims[1]
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(data_dir + images[i])
                original_img = img_to_array(original_img)/255
                x = random.randint(0, original_img.shape[0]-xw)
                y = random.randint(0, original_img.shape[1]-yw)
                croped_img = np.copy(original_img[x:x+xw, y:y+yw, :])

                imgs.append(croped_img)

                # masks
                if images[i].startswith("0"):
                    croped_mask = np.zeros((xw,yw))
                else:
                    original_mask = load_img(mask_dir + images[i].replace('full', 'mask'))
                    original_mask = img_to_array(original_mask)/255
                    x = random.randint(0, original_mask.shape[0]-xw)
                    y = random.randint(0, original_mask.shape[1]-yw)
                    croped_mask = original_mask[x:x+xw, y:y+yw, 0]
                labels.append(croped_mask)

            imgs = np.array(imgs)
            labels = np.array(labels)
            labels = labels.reshape(-1, dims[0], dims[1], 1)

            yield imgs, labels
