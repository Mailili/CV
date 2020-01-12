########################################################################################################
import cv2
import numpy as np

#替换证件照背景
img = cv2.imread("D:\Ksoftware\Anaconda\envs\cvProgramworks\imgs\\photo.jpg",1)
def background_replace():
    img_small_size = cv2.resize(img,(int(img.shape[0]*0.5),int(img.shape[1]*0.5)))
    #B,G,R = cv2.split(img_small_size)
    rows,cols,channels = img.shape
    #首先将BGR转为HSV格式，然后通过inRange函数获取背景的mask
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,0,225])
    upper_blue = np.array([180,5,255])                       #白色背景
    mask = cv2.inRange(hsv_img,lower_blue,upper_blue)
    cv2.namedWindow("dilate")
    cv2.namedWindow("mask")

    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyWindow()
    erode = cv2.erode(mask,None,iterations=1)
    cv2.imshow("mask",img)
    # cv2.imshow('erode',erode)
    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()
    dilate = cv2.dilate(erode,None,iterations=1)      #先erode再dilate 消除白点
    cv2.imshow("dilate", dilate)

    #更换背景图

    for i in range(rows):
        for j in range(cols):
            if dilate[i,j] == 255:
                img[i,j] = (0,255,255)
    cv2.imshow("img",img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyWindow()

background_replace()