import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

#读入彩色图，将彩色图转换为灰度图
def gray(imagePath):
    image=cv.imread(imagePath)
    imageGray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    imageGray=cv.resize(imageGray,(200,200))
    plt.imshow(imageGray,cmap="gray")
    plt.show()
    return imageGray

#利用Sobel算子卷积操作得到边缘检测结果
def Sobel(imageArray):
    sobelHorizontal=[[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelVertical=[[-1,-2,-1],[0,0,0],[1,2,1]]
    imageSize=imageArray.shape
    imageHeight=imageSize[0]
    imageWidth=imageSize[1]
    imageSobelHorizontal=np.zeros((imageHeight-2,imageWidth-2))
    imageSobelVertical=np.zeros((imageHeight-2,imageWidth-2))
    imageSobel=np.zeros((imageHeight-2,imageWidth-2))
    for i in range(imageHeight-2):
        for j in range(imageWidth-2):
            convolutionLine1=sobelHorizontal[0][0]*imageArray[i][j]+sobelHorizontal[0][1]*imageArray[i][j+1]+sobelHorizontal[0][2]*imageArray[i][j+2]
            convolutionLine2=sobelHorizontal[1][0]*imageArray[i+1][j]+sobelHorizontal[1][1]*imageArray[i+1][j+1]+sobelHorizontal[1][2]*imageArray[i+1][j+2]
            convolutionLine3=sobelHorizontal[2][0]*imageArray[i+2][j]+sobelHorizontal[2][1]*imageArray[i+2][j+1]+sobelHorizontal[2][2]*imageArray[i+2][j+2]
            imageSobelHorizontal[i][j]=convolutionLine1+convolutionLine2+convolutionLine3
            convolutionLine1=sobelVertical[0][0]*imageArray[i][j]+sobelVertical[0][1]*imageArray[i][j+1]+sobelVertical[0][2]*imageArray[i][j+2]
            convolutionLine2=sobelVertical[1][0]*imageArray[i+1][j]+sobelVertical[1][1]*imageArray[i+1][j+1]+sobelVertical[1][2]*imageArray[i+1][j+2]
            convolutionLine3=sobelVertical[2][0]*imageArray[i+2][j]+sobelVertical[2][1]*imageArray[i+2][j+1]+sobelVertical[2][2]*imageArray[i+2][j+2]
            imageSobelVertical[i][j]=convolutionLine1+convolutionLine2+convolutionLine3
    imageSobel=np.sqrt(imageSobelHorizontal**2+imageSobelVertical**2)
    plt.imshow(imageSobelVertical,cmap="gray")
    plt.show()
    plt.imshow(imageSobelHorizontal,cmap="gray")
    plt.show()
    plt.imshow(imageSobel,cmap="gray")
    plt.show()
    return imageSobel

#记录初始时间
beginTime=time.time()
#调用灰度化函数
imageArray=gray("cat0.jpg")
#调用Sobel边缘检测函数
Sobel(imageArray)
#记录结束时间
endTime=time.time()
#计算出运行时间
runningTime=endTime-beginTime
print("running time:",runningTime)