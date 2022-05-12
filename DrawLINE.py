import cv2
import os
import glob

num = 1
# sorted(glob.glob('./data/images/*.bmp'), key=lambda x:os.path.basename(x).split('.')[0],reverse=True):
for filePath in glob.glob('./data/images/*.bmp'):
    img_gray = cv2.imread(filePath, 0)
    if num ==1:
        cv2.line(img_gray,(3000,512),(3000,1000),(0,0,0),8)
    elif num ==2:
        cv2.line(img_gray,(3000,20),(3000,1000),(0,0,0),5)
    elif num ==3:
        cv2.line(img_gray, (3000, 20), (3000, 500), (0, 0, 0), 5)
        cv2.line(img_gray, (3000, 700), (3000, 1000), (0, 0, 0), 5)
    elif num ==4:
        cv2.line(img_gray, (3000, 20), (3000, 300), (0, 0, 0), 5)
    # elif num ==5:
    #     cv2.line(img_gray, (3500, 0), (3500, 1023), (0, 0, 0), 5)
    # elif num ==6:
    #     cv2.line(img_gray, (3500, 0), (3500, 400), (0, 0, 0), 5)
    # cv2.namedWindow(f'{num}',0)
    # cv2.imshow(f'{num}',img_gray)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('./data/images_long',f'{num}.bmp'), img_gray)
    num += 1