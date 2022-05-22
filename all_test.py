import os

import cv2
import time
import numpy as np
from PIL import Image

img_path = './result_roi_8k/SCILRTB_75_1_7100_992_6984_25_61_0_61_6919_6955_1_62_689_697_1452455_1452467_H.bmp'
# while 1:
#     time_1 = time.time()
#     img_gray = cv2.imread(img_path,0)
#     time_2 = time.time()
#     img_gray_1 = cv2.imread(img_path,1)
#     img_rgb = cv2.cvtColor(img_gray_1,cv2.COLOR_BGR2RGB)
#     time_3 = time.time()
#     print('gray: ',time_2 - time_1)
#     print('BGR: ',time_3-time_2)
#     time.sleep(1)
img_gray = cv2.imread(img_path,0)
img_pil = Image.fromarray(img_gray)
print(img_pil.size)
# img_arr = np.array(img_pil)
# print(img_arr.shape,img_arr.dtype)
# print(img_gray.shape,img_gray.dtype)
# img_rgb = np.stack([img_gray,img_gray,img_gray],axis=2)
# print(img_rgb.shape,img_rgb.dtype)
# cv2.namedWindow('1',0)
# cv2.imshow('1',img_gray)
# cv2.namedWindow('2',0)
# cv2.imshow('2',img_rgb)
# cv2.waitKey(0)