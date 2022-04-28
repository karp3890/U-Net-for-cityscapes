import cv2
import numpy as np
from tensorflow.python.keras.utils.all_utils import normalize, to_categorical
# from matplotlib import pyplot as plt
# image = cv2.imread("C:/projects/cnn_project/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")
# IMG_HEIGHT = 572
# IMG_WIDTH = 572
# image=cv2.resize(image,(IMG_HEIGHT,IMG_WIDTH))
# def imshow(image_path: str , size: int=10):
#     image=cv2.imread(image_path)
#     w, h =image.shape[0], image.shape[1]
#     aspect_ratio= w/h
#     plt.figure(figsize=(size*aspect_ratio,size))
#     plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#     plt.show()
#
# #cv2.imwrite("./output.jpg",image) # saving
# #imshow("C:/projects/cnn_project/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# #plt.show()
#
# print(image.shape)

# print(train_masks[0].shape)
# a=np.array(train_masks[0])
# a=a.reshape(-1,1)
#
#
# np.savetxt("./test.csv", a, delimiter=",")
# print(np.unique(train_masks[0]))
#
#


# train_masks = (glob.glob(TRAIN_PATH + '*/*/*.png', recursive=True))
# train_masks = list(map(lambda img: cv2.imread(img), train_masks))
# train_masks = list(map(lambda img: cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)), train_masks))
#
# test_masks = glob.glob(TEST_PATH +'*/*/*.png', recursive=True
# train_masks = list(map(lambda img: cv2.imread(img), train_masks))
# train_masks = list(map(lambda img: cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)), train_masks))
# valid_masks = glob.glob(VALID_PATH +'*/*/*.png', recursive=True)


# plt.imshow(cv2.cvtColor(train_masks[0], cv2.COLOR_BGR2RGB))
# plt.show()
a = to_categorical([0, 1, 2, 3], num_classes=4)
print(np.shape(a))
