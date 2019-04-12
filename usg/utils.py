import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import PIL
import os

import wget
import seaborn as sns
import scipy.signal as ss
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

import sys
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import shutil

from sklearn.model_selection import train_test_split

def raw_to_Bmode(raw, axis=0):
    """Transforms raw US to B-mode along given axis
    raw: 2D ndarray od raw US signal
    axis:
    """
    return 20* np.log10(np.abs(ss.hilbert(raw, axis=axis)))

def normalized(x, low, high):
    if not low:
        low = np.min(x)
    if not high:
        high = np.max(x)

    _x = np.copy(x)
    _x[_x<low] = low
    _x[_x>high] = high

    return (_x-low)/(high-low)

def normed_2_grayscale(x):
    return cm.gist_gray(x, bytes=True)

def nparray_2_PIL_image(x):
    return PIL.Image.fromarray(x)

def generate_names(path, ID, Class):
    name1 = "photos/%d_%s_%03d.%d.png"%(Class, "full", ID, 1)
    name2 = "masks/%d_%s_%03d.%d.png"%(Class, "mask", ID, 1)
    name3 = "photos/%d_%s_%03d.%d.png"%(Class, "full", ID, 2)
    name4 = "masks/%d_%s_%03d.%d.png"%(Class, "mask", ID, 2)
    names = [name1, name2, name3, name4]
    names = [os.path.join(path, name) for name in names]
    return names


def get_data_from_array(array, low=30, high=90):

    patien_ID = array[0][0]
    BI_RADS_category = array[5][0]
    Class = int(array[6][0][0])

    rf1 = normalized(raw_to_Bmode(array[1]), low, high)
    roi1 = array[3]

    rf2 = normalized(raw_to_Bmode(array[2]), low, high)
    roi2 = array[4]

    return (rf1, roi1), (rf2, roi2), (patien_ID, BI_RADS_category, Class)

def generate_photos(data, path):

    for i, sample in enumerate(tqdm(data)):
        (US1, mask1), (US2, mask2), (patien_ID, BI_RADS_category, Class) = get_data_from_array(sample)

        names = generate_names(path, i, Class)

        US1 = nparray_2_PIL_image(normed_2_grayscale(US1))
        US1.save(names[1])

        US2 = nparray_2_PIL_image(normed_2_grayscale(US2))
        US2.save(names[3])

def make_path(path):
    os.makedirs(path, exist_ok=True)


def reconstruct(data, path):
    make_path(path+'/masks')
    make_path(path+'/photos')


    for i, sample in enumerate(tqdm(data)):
        (US1, mask1), (US2, mask2), (patien_ID, BI_RADS_category, Class) = get_data_from_array(sample)

        names = generate_names(path, i, Class)

        US1 = nparray_2_PIL_image(US1)
        US1.save(names[0])

        US2 = nparray_2_PIL_image(US2)
        US2.save(names[2])

        mask1 = nparray_2_PIL_image(mask1*255)
        mask1.save(names[1])

        mask2 = nparray_2_PIL_image(mask2*255)
        mask2.save(names[3])