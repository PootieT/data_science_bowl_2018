import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import *
from skimage.morphology import label


def data_aug(image,label,angel=30,resize_rate=0.9):
  flip = random.randint(0, 1)
  size = image.shape[0]
  rsize = random.randint(np.floor(resize_rate*size),size)
  w_s = random.randint(0,size - rsize)
  h_s = random.randint(0,size - rsize)
  sh = random.random()/2-0.25
  rotate_angel = random.random()/180*np.pi*angel
  # Create Afine transform
  afine_tf = AffineTransform(shear=sh,rotation=rotate_angel)
  # Apply transform to image data
  image = warp(image, inverse_map=afine_tf,mode='reflect')
  label = warp(label, inverse_map=afine_tf,mode='reflect')
  # Randomly corpping image frame
  image = image[w_s:w_s+size,h_s:h_s+size,:]
  label = label[w_s:w_s+size,h_s:h_s+size]
  # Ramdomly flip frame
  if flip:
    image = image[:,::-1,:]
    label = label[:,::-1]
  return image, label


def resize_image(image):
  pad_width = 256 - np.shape(image)[1] % 256 if np.shape(image)[1] != 256 else 0
  pad_height = 256 - np.shape(image)[0] % 256 if np.shape(image)[0] != 256 else 0
  new_image = np.pad(image, ((0,pad_height), (0,pad_width), (0,0)), mode='reflect')
  return new_image

def break_image(image):
  images = []
  num_col = int(np.shape(image)[0] / 256)
  num_row = int(np.shape(image)[1] / 256)
  for i in range(num_col):
    for j in range(num_row):
      images.append(image[i*256:(i+1)*256,j*256:(j+1)*256,:])
  return images

def make_images(images, shape):
  num_col = int(shape[0] / 256 + 1) if shape[0] != 256 else 0
  num_row = int(shape[1] / 256 + 1) if shape[1] != 256 else 0
  new_image = np.zeros((num_col*256, num_row*256, shape[2]))
  for i in range(num_col):
    for j in range(num_row):
      new_image[i*256:(i+1)*256,j*256:(j+1)*256,:] = images[i*num_row+j]
  return new_image[:shape[0], :shape[1],:]

def rle_encoding(x):
  dots = np.where(x.T.flatten() == 1)[0]
  run_lengths = []
  prev = -2
  for b in dots:
    if (b>prev+1): run_lengths.extend((b + 1, 0))
    run_lengths[-1] += 1
    prev = b
  return run_lengths

def prob_to_rles(x, cutoff=0.5):
  lab_img = label(x > cutoff)
  for i in range(1, lab_img.max() + 1):
    yield rle_encoding(lab_img == i)
