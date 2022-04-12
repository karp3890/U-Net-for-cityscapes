from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import Cropping2D
from constants import *
from matplotlib import pyplot as plt
import cv2
image = cv2.imread("C:/projects/cnn_project/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")
IMG_HEIGHT = 572
IMG_WIDTH = 572
image=cv2.resize(image,(IMG_HEIGHT,IMG_WIDTH))
def imshow(image_path: str , size: int=10):
    image=cv2.imread(image_path)
    w, h =image.shape[0], image.shape[1]
    aspect_ratio= w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()


def double_conv(depth, input_layer, padding, name):
    convolution_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                               kernel_initializer=KERNEL_INITIALIZER,
                               padding=PADDING)(
        input_layer)
    print(f"{name} shape after 1st convolution:\n {convolution_layer.shape}")
    output_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION, kernel_initializer=KERNEL_INITIALIZER,
                          padding=PADDING)(
        convolution_layer)
    print(f"{name} shape after 2nd convolution: \n {output_layer.shape}")
    return output_layer


def conv_and_d_pool_step(depth, input_layer, padding, name):
    max_pooling_layer = MaxPooling2D(POOLING)(input_layer)
    print(f"{name} shape after max pooling:\n{max_pooling_layer.shape}")
    convolution_layer = Conv2D(filters=depth, kernel_size=KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                               kernel_initializer=KERNEL_INITIALIZER, padding=padding)(max_pooling_layer)
    print(f"{name} shape after 1st convolution:\n {convolution_layer.shape}")
    output_layer = Conv2D(filters=depth, kernel_size=KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                          kernel_initializer=KERNEL_INITIALIZER, padding=PADDING)(convolution_layer)

    print(f"{name} shape after 2nd convolution:\n {output_layer.shape}")
    return output_layer


def m_pool_con_conv_step(depth, input_layer, encoder_layer, padding,crop,name):
    down_pool_layer = Conv2DTranspose(depth, KERNEL_SIZE, strides=STRIDES, padding=padding)(input_layer)
    print(f"{name} shape after DOWN pooling:\n{down_pool_layer.shape}")
    print(down_pool_layer.shape)
    encoder_layer = Cropping2D(crop)(encoder_layer)
    concatenation_layer = concatenate([down_pool_layer, encoder_layer])
    print(f"{name} shape after concatenation:\n {concatenation_layer.shape}")
    convolution_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                               kernel_initializer=KERNEL_INITIALIZER, padding=PADDING)(concatenation_layer)
    print(f"{name} shape after 1st convolution:\n {convolution_layer.shape}")
    output_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION, kernel_initializer=KERNEL_INITIALIZER,
                          padding=PADDING)(convolution_layer)
    print(f"{name} shape after 2nd convolution:\n {output_layer.shape}")
    return output_layer


def unet_model():
    input = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # (572,572,3) => (568,68,64)
    first_encoder_layer = double_conv(64, input, VALID, "FIRST_ENCODER_LAYER")
    #  (568,68,64)=> (280, 280, 128)
    second_encoder_layer = conv_and_d_pool_step(128, first_encoder_layer, VALID, "SECOND_ENCODER_LAYER")
    # (140,140,128) =>  (68, 68, 256)
    third_encoder_layer = conv_and_d_pool_step(256, second_encoder_layer, VALID, "THIRD_ENCODER_LAYER")
    # (68,68,256) => ( 32, 32, 512)
    fourth_encoder_layer = conv_and_d_pool_step(512, third_encoder_layer, VALID, "FOURTH_ENCODER_LAYER")
    # # ( 32, 32, 512) =>  (28, 28, 1024)
    middle_layer = conv_and_d_pool_step(1024, fourth_encoder_layer, VALID, "MIDDLE_LAYER")

    fourth_decoder_layer = m_pool_con_conv_step(512, middle_layer, fourth_encoder_layer, SAME,(4,4), "FOURTH_DECODER_LAYER")
    third_decoder_layer = m_pool_con_conv_step(256,fourth_decoder_layer, third_encoder_layer,SAME,(16,16), "THIRD_DECODER_LAYER")
    second_decoder_layer = m_pool_con_conv_step(128,third_decoder_layer,second_encoder_layer,SAME,(40,40), "SECOND_DECODER_LAYER")
    first_decoder_layer = m_pool_con_conv_step(64,second_decoder_layer,first_encoder_layer,SAME, (88,88), "FIRST_DECODER_LAYER")
    out_put_layer = Conv2D( CLASS_NUMBER, (1,1), activation=ACTIVATION_FUNCTION,
                           kernel_initializer=KERNEL_INITIALIZER, padding=PADDING)(first_decoder_layer)
    print(out_put_layer.shape)
    #

    # # Expansive path
    # u6 = Conv2DTranspose(512, KERNEL_SIZE, strides=STRIDES, padding=PADDING)(c5)
    # u6 = concatenate([u6, c4])
    # c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    #
    # c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    #

    # u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    # u7 = concatenate([u7, c3])
    # c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    #
    # c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    #
    # u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = concatenate([u8, c2])
    # c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    #
    # c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    #
    # u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    # u9 = concatenate([u9, c1], axis=3)
    # c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    #
    # c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    output = Conv2D(CLASS_NUMBER, (1, 1), activation='softmax')(input)
    return output


unet_model()


def build(input, output):
    model = Model(inputs=[input], outputs=[output])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

# NOTE: Compile the model in the main program to make it easy to test with various loss functions
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()
