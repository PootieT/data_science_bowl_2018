import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from data_utils import resize_image, break_image, make_images, prob_to_rles

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

cluster = '2'
threshold = 0.5

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TEST_PATH = 'data/stage1_test/'


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# Define IoU metric, according to kaggle this isnt correct
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)



test_ids = next(os.walk(TEST_PATH))[1]

X_test_no_resize = [0]*len(test_ids)
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    X_test_no_resize[n] = img

cluster_dict = {}
with open('clusters_test.txt','r') as f:
    for line in f.readlines():
        temp = line.split(',')
        if temp[1] != 'image_id':
            cluster_dict[temp[1]] = temp[2].strip()

masks_dict = {}
model = load_model('model-cluster-'+ cluster +'-0.h5', custom_objects={'mean_iou': mean_iou})
new_test_ids = []
rles = []


for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_cluster = cluster_dict[id_]
    if img_cluster != cluster:
        continue
    original_shape = np.shape(X_test_no_resize[n])
    images = break_image(resize_image(X_test_no_resize[n]))
    masks = []
    for image in images:
        temp_image = np.reshape(np.expand_dims(image,axis=-1), (1,) + np.shape(image))
        preds_val = model.predict_on_batch(temp_image)
        masks.append(preds_val)
    mask = make_images(masks, (original_shape[0], original_shape[1],1))
    rle = list(prob_to_rles(mask, cutoff = threshold))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-cluster-'+ cluster +'-1.csv', index=False)
