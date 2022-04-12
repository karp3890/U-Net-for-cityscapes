import cv2
import numpy as np
from matplotlib import pyplot as plt
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

#cv2.imwrite("./output.jpg",image) # saving
#imshow("C:/projects/cnn_project/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#plt.show()

print(image.shape)