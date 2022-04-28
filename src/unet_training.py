from constants import *
from utils import *
from tensorflow.python.keras.utils.all_utils import normalize, to_categorical
import glob
import sys
import numpy as np
from unet_model import Unet
from matplotlib import pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
masks = convert_images(TRAIN_MASK_PATH, MASK_MODE, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
masks = label_encoder(masks)

images = convert_images(TRAIN_IMG_PATH, IMAGE_MODE, (INPUT_HEIGHT, INPUT_WIDTH))
images = normalize(images, axis=1)


data = get_subsets(masks_to_categorical(masks), images)


model =Unet()
model.run(data['image_test'], data['masks_test'])