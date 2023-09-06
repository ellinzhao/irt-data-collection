import os

import cv2
import numpy as np
from PIL import Image
from scipy.constants import convert_temperature as conv_temp


DATETIME_TAG = 36867
tform = np.genfromtxt('./tform.txt', delimiter=', ')


# ######################## IR-RGB processing #########################

def coords_rgb_to_ir(pts):
    x = np.hstack((np.array(pts), np.ones((len(pts), 1))))
    y = np.dot(tform, x.T)
    return y[:2].astype(int)


# ############################# File I/O ##############################

rgb2gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
bgr2rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def raw2temp(raw, units='F'):
    return conv_temp((raw - 27315) / 100, 'C', units)


def raw2viz(raw):
    # To make the raw images "previewable", convert to 8bit data
    bit = cv2.normalize(raw, None, 0, 65535, cv2.NORM_MINMAX)
    bit = np.right_shift(bit, 8)
    return bit.astype(np.uint8)


def load_im(path, transform=None):
    if not transform:
        transform = lambda raw: raw
    return transform(cv2.imread(path, -1))


def load_dir(dir, transform=None):
    # Load all raw images in the directory
    # `transform` should be function

    n = len(os.listdir(dir))
    dir_ims = []
    for i in range(n):
        path = os.path.join(dir, f'im{i}.png')
        raw = load_im(path, transform)
        dir_ims.append(transform(raw))
    return np.array(dir_ims)


def read_timestamp(path):
    im = Image.open(path)
    exif = im.getexif()
    return exif[DATETIME_TAG]


# ############################# Miscellaneous ##############################

def adjust_gamma(im, gamma=1.0):
    # Uses lookup table for faster performance
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0)**invGamma) * 255 for i in range(256)
    ]).astype('uint8')
    return cv2.LUT(im, table)
