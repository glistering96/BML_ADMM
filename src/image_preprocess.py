import pandas as pd
import numpy as np
import os
from natsort import natsorted, ns
from skimage import io, color
import skimage
import multiprocessing as mp
from threading import Thread

DATA_SRC = "../data/coil-100/"
DATA_TARGET = "../data/gray/"
list_files = os.listdir(DATA_SRC)

def gray(filename):
    original = io.imread(DATA_SRC + filename)
    img = color.rgb2gray(original)
    new_name = filename.split(".")
    io.imsave(DATA_TARGET + f"{new_name[0]}.png", img)
    return img

def save(img):
    io.imsave(DATA_SRC + "gray/", img)

if __name__ == '__main__':
    # pool = mp.Pool(6)
    #
    # result = pool.map(gray, list_files)

    for f in list_files:
        gray(f)

