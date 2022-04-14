from tensorflow.python.keras.layers import Input
IMG_HEIGHT: int = 572
IMG_WIDTH: int = 572
IMG_CHANNELS: int = 3
CLASS_NUMBER: int = 4
ACTIVATION_FUNCTION: str = "relu"
KERNEL_INITIALIZER: str = "he_normal"  # wagi poczatkowe
PADDING: str = "valid" # add paddding to the borders
KERNEL_SIZE = (3, 3)
POOLING = (2,2)
DEPTHS = (64,128,256,512,1024)
STRIDES = (2,2)
VALID="valid"
SAME="same"
CROP=(4,4)
INPUT = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
INPUT_SHAPE=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)