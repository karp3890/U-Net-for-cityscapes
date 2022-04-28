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


def get_subsets(masks, images, test_size=DEFAULT_TEST_SIZE):
    subsets = train_test_split(masks, images, train_size=test_size)
    subsets_dict = {"masks_test": subsets[0],
                    "masks_train": subsets[1],
                    "image_test": subsets[2],
                    "image_train": subsets[3]
                    }
    return subsets_dict


def convert_images(path, mode, size):
    dataset = (glob.glob(path + '*/*/*.png', recursive=True))
    dataset = list(map(lambda img: cv2.imread(img, mode), dataset))
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


def masks_to_categorical(mask_dataset):
    target_shape = np.shape(mask_dataset) + (CLASS_NUMBER,)
    print(f"Masks target shape: {target_shape}")
    mask_dataset = np.ravel(mask_dataset)
    mask_dataset_categorical = to_categorical(mask_dataset, CLASS_NUMBER)
    mask_dataset_categorical_reshaped = np.reshape(mask_dataset_categorical, target_shape)
    print(f"Masks final shape: {np.shape(target_shape)}")
    return mask_dataset_categorical_reshaped
