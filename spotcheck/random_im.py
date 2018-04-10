import os
import numpy as np

for index in np.random.randint(0,48000,20):
  os.system("scp ../data/stage1_train_aug_6/cluster_all/images/image_"+str(index)+".png image_" + str(index) + ".png")
  os.system("scp ../data/stage1_train_aug_6/cluster_all/masks/mask_"+str(index)+".png  mask_" + str(index) + ".png")

