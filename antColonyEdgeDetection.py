import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time

#读入彩色图，将彩色图转换为200*200大小的灰色图，并返回灰度矩阵
def preProcessImage(filePath):
    imageOriginal=cv.imread(filePath)
    imageResize=cv.resize(imageOriginal,(200,200))
    imageGray=cv.cvtColor(imageResize,cv.COLOR_BGR2GRAY)
    # cv.imshow("pic",imageGray)
    # cv.waitKey(0)
    return imageGray

#计算一个像素周围像素的灰度差距，输入这个像素的（X，Y）坐标，返回其8邻域的灰度差距数组
def distanceCalculation(pixelX,pixelY,imageGray):
    distance=np.zeros(8)
    pixelX=int(pixelX)
    pixelY=int(pixelY)
    step=1
    distance[0]=math.fabs(int(imageGray[pixelX-step][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[1]=math.fabs(int(imageGray[pixelX][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[2]=math.fabs(int(imageGray[pixelX+step][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[3]=math.fabs(int(imageGray[pixelX-step][pixelY])-int(imageGray[pixelX][pixelY]))
    distance[4]=math.fabs(int(imageGray[pixelX+step][pixelY])-int(imageGray[pixelX][pixelY]))
    distance[5]=math.fabs(int(imageGray[pixelX-step][pixelY+step])-int(imageGray[pixelX][pixelY]))
    distance[6]=math.fabs(int(imageGray[pixelX][pixelY+step])-int(imageGray[pixelX][pixelY]))
    distance[7]=math.fabs(int(imageGray[pixelX+step][pixelY+step])-int(imageGray[pixelX][pixelY]))
    return distance

#初始化
def initialize(antNum,imageGray):
    #设置计数器，如果该点与其8邻域的像素点没有灰度差距，则重新取点
    count=0
    distance=np.zeros(8)
    startingPoint = np.zeros(2)
    while(count==0):
        # 随机生成一个起始点
        startingPoint[0] = np.random.randint(20, 180, 1)
        startingPoint[1] = np.random.randint(20, 180, 1)
        startingPoint = np.array(startingPoint)
        distance=distanceCalculation(startingPoint[0],startingPoint[1],imageGray)
        for i in range(8):
            if distance[i]>20:
                count+=1
    #信息素矩阵，行为蚂蚁数（走到的像素），每行为每只蚂蚁（像素）周围的8邻域信息素
    pheromone=np.ones((antNum,8))
    #蚂蚁路径，行为蚂蚁数（走到的像素），每行为每只蚂蚁（像素）的下一个移动坐标
    antRoute=np.zeros((antNum,2))
    return pheromone,antRoute,startingPoint

#算出8邻域各个位置的概率
def probabilityCalculation(antIndex,distance,pheromone,alpha,beta):
    probability=np.zeros(8)
    for i in range(8):
        probability[i]=pheromone[antIndex][i]**alpha+distance[i]**beta
    probabilitySum=np.sum(probability)
    probability=probability/probabilitySum
    return probability

#轮盘赌选择决定8个邻域中的下一个访问像素
def roulette(probability):
    probabilityTotal = np.zeros(len(probability))
    probabilityTmp = 0
    for i in range(len(probability)):
        probabilityTmp += probability[i]
        probabilityTotal[i] = probabilityTmp
    randomNumber=np.random.rand()
    result=0
    for i in range(1, len(probabilityTotal)):
        if randomNumber<probabilityTotal[0]:
            result=0
            break
        elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
            result=i
    return result

#记录第i个像素到第i+1个像素之间的变化
def singleTransfer(antIndex,startingPoint,pheromone,antRoute,imageGray,alpha,beta,rho):
    #记录下第i（antIndex）个像素的坐标
    antRoute[antIndex][0]=startingPoint[0]
    antRoute[antIndex][1]=startingPoint[1]
    #计算第i（antIndex)8邻域的灰度距离
    distance=distanceCalculation(startingPoint[0],startingPoint[1],imageGray)
    #计算概率
    probability=probabilityCalculation(antIndex,distance,pheromone,alpha,beta)
    #通过轮盘赌获得下一个访问的像素坐标
    nextPixel=roulette(probability)
    nextPoint=np.zeros(2)
    step= 1
    if nextPixel==0:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0]<step:
            nextPoint[0]=step
        elif nextPoint[0]>=197-step:
            nextPoint[0]=197-step
        if nextPoint[1]<step:
            nextPoint[1]=step
        elif nextPoint[1]>=197-step:
            nextPoint[1]=197-step
    elif nextPixel==1:
        nextPoint[0]=startingPoint[0]
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==2:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] =step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==3:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==4:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==5:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==6:
        nextPoint[0]=startingPoint[0]
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==7:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step

    #设置计数器，如果下一个点的周围都是没有灰度差异的，则一直维持该点不动
    count=0
    for i in range(8):
        if distance[i]<20:
            count+=1
    if count>7 or count<1:
        nextPoint=startingPoint
    else:
        # 更新信息素
        deltaPheromone = np.zeros(8)
        for i in range(8):
            if distance[i] > 20:
                deltaPheromone[i] = distance[i] / 255
        deltaPheromoneSum = np.sum(deltaPheromone)
        for i in range(8):
            pheromone[antIndex][i] = (1 - rho) * pheromone[antIndex][i] + deltaPheromoneSum
    return nextPoint, pheromone, antRoute
#记录若干个像素之间的变化
def singleIteration(antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho):
    pheromone_,antRoute_,point=initialize(antNum,imageGray)
    for i in range(antNum):
        point,pheromone,antRoute=singleTransfer(i,point,pheromone,antRoute,imageGray,alpha,beta,rho)
    return point,pheromone,antRoute

#若干次迭代
def severalIteration(iterateTimes,antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho):
    for i in range(iterateTimes):
        print("iterate ",i,":")
        point,pheromone,antRoute=singleIteration(antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho)
        draw(antRoute)
    plt.show()
    return point,pheromone,antRoute

#画出边缘路径
def draw(antRoute):
    plt.plot(antRoute[:, 0], antRoute[:, 1],color="blue")
    imageDetection = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            imageDetection[i][j] = 0
    for i in range(len(antRoute)):
        x = int(antRoute[i][0])
        y = int(antRoute[i][1])
        imageDetection[x][y] = 255

#设置参数
alpha=1
beta=1
rho=0.3
#记录算法开始时间
beginTime=time.time()
#灰度化图像
imageGray=preProcessImage("cat.jpg")
#初始化信息，选50只蚂蚁组成连续路径
pheromone,antRoute,startingPoint=initialize(100,imageGray)
#开始2000次迭代
point,pheromone,antRoute=severalIteration(2000,100,startingPoint,pheromone,antRoute,imageGray,alpha,beta,rho)
#将结果画出
draw(antRoute)
#记录算法结束时间
endTime=time.time()
#算出运行时间
runningTime=endTime-beginTime
print("running time:",runningTime)
