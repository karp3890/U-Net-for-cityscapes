import os.path

from tensorflow.python.keras.layers import Input

from keras.metrics import OneHotMeanIoU
DEFAULT_TEST_SIZE: float = 0.1
DEFAULT_VALIDATION_SPLIT: float = 0.3
VERBOSE = 1
CLASS_NUMBER: int = 7
DEFAULT_LOSS: str = "categorical_crossentropy"
DEFAULT_OPTIMIZER: str = "adam"
DEFAULT_METRICS = OneHotMeanIoU(CLASS_NUMBER)
DEFAULT_EPOCHS_SIZE: int = 100
DEFAULT_BATCH_SIZE: int = 1
INPUT_HEIGHT: int = 572
INPUT_WIDTH: int = 572
OUTPUT_HEIGHT: int = 388
OUTPUT_WIDTH: int = 388
IMG_CHANNELS: int = 3
ACTIVATION_FUNCTION: str = "relu"
KERNEL_INITIALIZER: str = "he_normal"  # wagi poczatkowe
PADDING: str = "valid"  # add paddding to the borders
KERNEL_SIZE = (3, 3)
POOLING = (2, 2)
DEPTHS = (64, 128, 256, 512, 1024)
STRIDES = (2, 2)
VALID = "valid"
SAME = "same"
CROP = (4, 4)
INPUT = Input((INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS))
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS)
BASE_PATH_MASK = "C:\projects\cnn_project\data\gtFine"
BASE_PATH_IMG = "C:\projects\cnn_project\data\leftImg8bit"
TEST_MASK_PATH = os.path.join(BASE_PATH_MASK, "test")
TRAIN_MASK_PATH = os.path.join(BASE_PATH_MASK, "train")
VALID_MASK_PATH = os.path.join(BASE_PATH_MASK, "val")
TEST_IMG_PATH = os.path.join(BASE_PATH_IMG, "test")
TRAIN_IMG_PATH = os.path.join(BASE_PATH_IMG, "train")
VALID_IMG_PATH = os.path.join(BASE_PATH_IMG, "val")

MASK_MODE = 0
IMAGE_MODE = 1
