import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import pylab

#operncv读取和打印操作
img = cv2.imread("D:\Ksoftware\Anaconda\envs\cvProgramworks\imgs\\timg.jpg",1)
def cvReadImshow(img,img_name = "mie"):

    cv2.imshow(img_name,img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

#matplotlib读取与打印   注意RGB通道不同！！！matplotlib是GBR
def   pltImshow(img):
    # plt.imshow(img)
    # pylab.show()
    #通道的操作
    B,G,R = cv2.split(img)
    img_rgb = cv2.merge((R,G,B))
    plt.imshow(img_rgb)
    pylab.show()
#pltImshow()

#图像的存储方式
# print(img.shape)
# print(img.dtype)
# print(img)


#图像裁剪
def imgCrop():
    img_crop = img[200:400,200:400]
    cv2.imshow("img_crop",img_crop)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


 #灰阶平移
def random_light_color(img):
    B,G,R = cv2.split(img)              #先分离通道

    b_rand = random.randint(-50,50)
    if b_rand == 0:
         pass
    elif b_rand > 0:
         lim = 255-b_rand
         B[B > lim] = 255
         B[B <= lim] = (b_rand+B[B <= lim]).astype(img.dtype)    #记得转换为图片类型
    elif b_rand < 0:
         lim = b_rand
         B[B < lim] = 0
         B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50,50)
    if g_rand == 0:
     pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim ] = (G[G<=lim] + g_rand).astype(img.dtype)
    elif g_rand < 0:
         lim = g_rand
         G[G<lim] = 0
         G[G>=lim] = (G[G>=lim]+ g_rand).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
         pass
    elif r_rand > 0:
         lim = 255 - r_rand
         R[R > lim] = 255
         R[R <= lim] = (R[R <= lim] + r_rand).astype(img.dtype)
    elif r_rand < 0:
         lim = r_rand
         R[R > lim] = 0
         R[R <= lim] = (R[R <= lim] + r_rand).astype(img.dtype)

    img_merge = cv2.merge((B,G,R))    #记住opencv的通道顺序是BGR  两层括号！
    return img_merge

#test
# img_lightcolor = random_light_color(img)
# pltImshow(img_lightcolor)


def gamma_adjust(img,gamma):       #gamma指数
    #cvReadImshow(img)
    #key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0)**invGamma)*255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img,table)

#test
#img_brighter = gamma_adjust(img,8)
#cvReadImshow(img_brighter)

#直方图操作  plt.hit
def hist_show():
    img_small_size = cv2.resize(img_brighter,(int(img_brighter.shape[0]*0.5),int(img_brighter.shape[1]*0.5)))
    plt.hist(img_small_size.flatten(),256,[0,256],color='r') #plt.hist(数据，条形数，x轴范围，颜色）
    plt.show()                      #展示原始直方图分布情况

    img_yuv = cv2.cvtColor(img_small_size,cv2.COLOR_BGR2YUV)    #颜色空间转换
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])           #equlize the Y channel
    img_output =cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
    cvReadImshow(img_output)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

#hist_show()

#相似/仿射/投影变换

#rotation
def imgRotate(rotate_angle,scale):
    M= cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rotate_angle,scale)
    img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    cvReadImshow(img_rotate)
    print(M)
#imgRotate(60,0.5)

def AffineTransform():
    row,col,channel = img.shape
    #三对点确定仿射变换
    pts1 = np.float32([[0,0],[col-1,0],[0,row-1]])
    pts2 = np.float32([[col*0.2,row*0.1],[col*0.9,row*0.2],[col*0.1,row*0.9]])
    M = cv2.getAffineTransform(pts1,pts2)           #注意此处仿射变换矩阵获取函数
    dst = cv2.warpAffine(img,M,(row,col))
    cvReadImshow(dst,"Affine")


#perspective transform                      #8对点确定投影变换
def random_warp(img):
    height,width,channels = img.shape
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])

    M_warp = cv2.getPerspectiveTransform(pts1,pts2)
    img_warp = cv2.warpPerspective(img,M_warp,(width,height))
    return M_warp,img_warp

#test
# M,img_warp = random_warp(img)
# cvReadImshow(img_warp)


#膨胀与腐蚀

# img1 = cv2.erode(img,None,iterations=1)
# cvReadImshow(img1)
# img2 = cv2.dilate(img,None,iterations=1)
# cvReadImshow(img2)