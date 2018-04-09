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

from data_utils import resize_image, break_image, data_aug

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'data/stage1_train/'
AUGMENT_PATH = 'data/stage1_train_aug_1/'
AUGS = 9


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
#seed = 42
#random.seed = seed
#np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]

X_train_no_resize = [0]*len(train_ids)
Y_train_no_resize = [0]*len(train_ids)
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  path = TRAIN_PATH + id_
  img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
  X_train_no_resize[n] = img
  mask_bool = 0
  for mask_file in next(os.walk(path + '/masks/'))[2]:
    mask_ = imread(path + '/masks/' + mask_file)
    mask_ = np.expand_dims(mask_, axis=-1)
    if mask_bool == 0:
      mask = np.zeros((mask_.shape[0], mask_.shape[1], 1), dtype=np.bool)
      mask_bool = 1
    mask = np.maximum(mask, mask_)
  Y_train_no_resize[n] = mask


cluster_dict = {}
with open('clusters.txt','r') as f:
  for line in f.readlines():
    temp = line.split(',')
    if temp[1] != 'image_id':
      cluster_dict[temp[1]] = temp[2].strip()


images_counter = {'0':0, '1':0, '2':0}
os.mkdir(AUGMENT_PATH)

os.mkdir(AUGMENT_PATH + 'cluster_0')
os.mkdir(AUGMENT_PATH + 'cluster_1')
os.mkdir(AUGMENT_PATH + 'cluster_2')

os.mkdir(AUGMENT_PATH + 'cluster_0/images')
os.mkdir(AUGMENT_PATH + 'cluster_0/masks')

os.mkdir(AUGMENT_PATH + 'cluster_1/images')
os.mkdir(AUGMENT_PATH + 'cluster_1/masks')

os.mkdir(AUGMENT_PATH + 'cluster_2/images')
os.mkdir(AUGMENT_PATH + 'cluster_2/masks')

for ix in range(len(train_ids)):
  if ix % 10 == 0: print(ix)
  cluster = cluster_dict[train_ids[ix]]
  path = AUGMENT_PATH + 'cluster_' + cluster
  images = break_image(resize_image(X_train_no_resize[ix]))
  masks = break_image(resize_image(Y_train_no_resize[ix]))
  for j in range(len(images)):
    imsave(path+'/images/image_'+str(images_counter[cluster])+'.png', images[j])
    imsave(path+'/masks/mask_'+str(images_counter[cluster])+'.png', np.squeeze(masks[j]))
    images_counter[cluster] += 1
  for i in range(AUGS):
    new_image, new_mask = data_aug(X_train_no_resize[ix], Y_train_no_resize[ix])
    images = break_image(resize_image(new_image))
    masks = break_image(resize_image(new_mask))
    for j in range(len(images)):
      imsave(path+'/images/image_'+str(images_counter[cluster])+'.png', images[j])
      imsave(path+'/masks/mask_'+str(images_counter[cluster])+'.png', np.squeeze(masks[j]))
      images_counter[cluster] += 1
print(images_counter)
