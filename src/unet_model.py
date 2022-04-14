from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import Cropping2D
from constants import *


class Unet(Model):

    def __init__(self):
        super().__init__()
        self.inputs= INPUT
        self.outputs = self.unet_model(INPUT)

    def unet_model(self,input_layer):
        # (572,572,3) => (568,68,64)
        first_encoder_layer = self.double_conv(64, input_layer, VALID, "FIRST_ENCODER_LAYER")
        #  (568,68,64)=> (280, 280, 128)
        second_encoder_layer = self.double_conv(128, first_encoder_layer, VALID, "SECOND_ENCODER_LAYER")
        # (140,140,128) =>  (68, 68, 256)
        third_encoder_layer = self.conv_and_d_pool_step(256, second_encoder_layer, VALID, "THIRD_ENCODER_LAYER")
        # (68,68,256) => ( 32, 32, 512)
        fourth_encoder_layer = self.conv_and_d_pool_step(512, third_encoder_layer, VALID, "FOURTH_ENCODER_LAYER")
        # # ( 32, 32, 512) =>  (28, 28, 1024)
        middle_layer = self.conv_and_d_pool_step(1024, fourth_encoder_layer, VALID, "MIDDLE_LAYER")

        fourth_decoder_layer = self.m_pool_con_conv_step(512, middle_layer, fourth_encoder_layer, SAME,(4,4), "FOURTH_DECODER_LAYER")

        third_decoder_layer = self.m_pool_con_conv_step(256,fourth_decoder_layer, third_encoder_layer,SAME,(16,16), "THIRD_DECODER_LAYER")

        second_decoder_layer = self.m_pool_con_conv_step(128,third_decoder_layer,second_encoder_layer,SAME,(40,40), "SECOND_DECODER_LAYER")

        first_decoder_layer = self.m_pool_con_conv_step(64,second_decoder_layer,first_encoder_layer,SAME, (88,88), "FIRST_DECODER_LAYER")

        output_layer = Conv2D( CLASS_NUMBER, (1,1), activation=ACTIVATION_FUNCTION,
                               kernel_initializer=KERNEL_INITIALIZER, padding=PADDING)(first_decoder_layer)
        return output_layer



    def conv_and_d_pool_step(self,depth, input_layer, padding, name):
        max_pooling_layer = MaxPooling2D(POOLING)(input_layer)
        print(f"{name} shape after max pooling:\n{max_pooling_layer.shape}")
        convolution_layer = Conv2D(filters=depth, kernel_size=KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                                   kernel_initializer=KERNEL_INITIALIZER, padding=padding)(max_pooling_layer)
        print(f"{name} shape after 1st convolution:\n {convolution_layer.shape}")
        output_layer = Conv2D(filters=depth, kernel_size=KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                              kernel_initializer=KERNEL_INITIALIZER, padding=padding)(convolution_layer)

        print(f"{name} shape after 2nd convolution:\n {output_layer.shape}")
        return output_layer


    def m_pool_con_conv_step(self,depth, input_layer, encoder_layer, padding,crop,name):
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

    def double_conv(self,depth, input_layer, padding, name):
        convolution_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION,
                                   kernel_initializer=KERNEL_INITIALIZER,
                                   padding=PADDING)(input_layer)
        print(f"{name} shape after 1st convolution:\n {convolution_layer.shape}")
        output_layer = Conv2D(depth, KERNEL_SIZE, activation=ACTIVATION_FUNCTION, kernel_initializer=KERNEL_INITIALIZER,
                              padding=PADDING)(convolution_layer)
        print(f"{name} shape after 2nd convolution: \n {output_layer.shape}")
        return output_layer
    def call(self,inputs):
        x= self.inputs(inputs)
        return self.outputs(x)
model = Unet()
# model.compile()
# model.summary()
model.unet_model(INPUT)


#     model = Model(inputs=[input], outputs=[output])
