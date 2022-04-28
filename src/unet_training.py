from constants import *
from tensorflow.python.keras.utils.all_utils import normalize, to_categorical
import glob
import sys
import numpy as np
from unet_model import Unet
from matplotlib import pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import constant

from matplotlib import pyplot as plt


# spark = SparkSession.builder.getOrCreate()

# result = map(lambda x: x + x, numbers)
def convert_images(path,mode,size):
    dataset = (glob.glob(path + '*/*/*.png', recursive=True))
    dataset = list(map(lambda img: cv2.imread(img,mode), dataset))
    dataset = list(map(lambda img: cv2.resize(img, size, interpolation=cv2.INTER_NEAREST), dataset))
    dataset = np.array(dataset)
    return dataset


"""
Function below converts categories' values to be counted from 0 to n, where n is a number of categories
"""
def label_encoder(dataset):
    label = LabelEncoder()
    dataset_original_shape = dataset.shape
    dataset = dataset.reshape(-1, 1)
    dataset = label.fit_transform(dataset)
    dataset = dataset.reshape(dataset_original_shape)
    return dataset


train_masks = convert_images(TRAIN_MASK_PATH,MASK_MODE,(OUTPUT_HEIGHT,OUTPUT_WIDTH))

train_masks = label_encoder(train_masks)


train_img = convert_images(TRAIN_IMG_PATH, IMAGE_MODE, (INPUT_HEIGHT, INPUT_WIDTH))

train_img =normalize(train_img,axis=1)

masks_test,masks_train,img_test,img_train = train_test_split(train_masks, train_img, train_size=0.1)
# print(np.shape(masks_test))
# print(np.shape(img_test))
print(np.unique(masks_test))

##TO USE WITH CATEGORICAL CROSS ENTROPY####
print(np.shape(masks_test))
masks_test_shape=np.shape(masks_test)
masks_test=np.ravel(masks_test)
print(np.shape(masks_test))
masks_test_categorical = to_categorical(masks_test, num_classes=CLASS_NUMBER)
print(np.shape(masks_test_categorical))
masks_test_categorical = np.reshape(masks_test_categorical,(masks_test_shape)+(CLASS_NUMBER,))
print(np.shape(masks_test_categorical))

model = Unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
model
model.model(masks_test_categorical,img_test)
#






# masks_test_categorical = masks_test_categorical.reshape(masks_test.shape[0],masks_test.shape[1],masks_test.shape[2],CLASS_NUMBER)
# print(np.shape(masks_test_categorical))
#




# np.set_printoptions(threshold=sys.maxsize)
#



