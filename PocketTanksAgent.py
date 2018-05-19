###############################################################################
#Imports
###############################################################################
import pandas as pd
import cv2
import numpy as np
from pywinauto import application
from time import time, sleep
import win32api, win32con
from math import sqrt

from keras.models import Sequential, load_model
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from keras import optimizers
###############################################################################
#OpenCV
###############################################################################
lower_tank1 = np.array([30,50,0])
upper_tank1 = np.array([35,255,255])
lower_tank2 = np.array([75,0,0])
upper_tank2 = np.array([110,255,255])
lower_world = np.array([50,0,0])
upper_world = np.array([70,255,255])
lower_explosion = np.array([135,100,150])
upper_explosion = np.array([140,255,255])

yellowTankTemplate = cv2.imread('tank_yellow.png', 0)
blueTankTemplate = cv2.imread('tank_blue.png', 0)

yellowTankTemplate2 = cv2.imread('tank_yellow3.png', 0)
blueTankTemplate2 = cv2.imread('tank_blue3.png', 0)
orb = cv2.ORB_create()

kpYellowTank, desYellowTank = orb.detectAndCompute(yellowTankTemplate, None)
kpBlueTank, desBlueTank = orb.detectAndCompute(blueTankTemplate, None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)


def detectWorldLowestAngle(yellowTankCoordinates, blueTankCoordinates):
    #Taking a snapshot
    img = np.array(app["Pocket Tanks Deluxe"].CaptureAsImage())
    #OpenCV processing
    img = img[31+33:rows, 10:cols]    #31 for the titlebar and 33 for the names
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_world, upper_world)
    resWorld = cv2.bitwise_and(img, img, mask = mask)
    gray = cv2.cvtColor(resWorld, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    #Corner detection
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
    if corners is not None:
        corners = np.int0(corners)
        
        l1, l2=[], []
        corners = list(corners)
        #Remove unnecessary corners
        i=0
        while i<len(corners):
            corner = corners[i]
            if corner[0][0]<yellowTankCoordinates[0] or corner[0][0]>blueTankCoordinates[0]:
                corners.pop(i)
                continue
            i+=1
        #Calculate angles
        for corner in corners:
            if corner[0][0] != yellowTankCoordinates[0]:
                a1 = np.arctan((yellowTankCoordinates[1] - corner[0][1])\
                               /(corner[0][0] - yellowTankCoordinates[0]))*180/np.pi
            if corner[0][0] != blueTankCoordinates[0]:
                a2 = -np.arctan((blueTankCoordinates[1] - corner[0][1])\
                               /(corner[0][0] - blueTankCoordinates[0]))*180/np.pi
            l1.append(a1)
            l2.append(a2)
        #Return the max angle
        if len(l1)==0 and len(l2)==0:
            return 0
        elif len(l1)==0:
            return max(l2)
        elif len(l2)==0:
            return max(l1)
        else:
            return max(max(l1), max(l2))
    else:
        return 0


def detectCollisionCoordinates():
    bg = cv2.createBackgroundSubtractorKNN()
    entryTime = time()
    counterOld = 0
    counterCurrent = 0
    
    #FIRE
    
    
    while True:
        try:
            img = np.array(app["Pocket Tanks Deluxe"].CaptureAsImage())
            #CROPPING AND CONVERTING IMAGES
            img = img[31+33:rows, 10:cols]    #31 for the titlebar and 33 for the names
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_explosion, upper_explosion)
            resExplosion = cv2.bitwise_and(img, img, mask = mask)
            
            mask = bg.apply(resExplosion)
            #DETECTING PROJECTILE BY CORNER DETECTION
            corners = cv2.goodFeaturesToTrack(mask, 50, 0.01, 20)
            if corners is not None:
                counterCurrent += 1
                explosionPointsX = []
                explosionPointsY = []
                corners = np.int0(corners)
                for corner in corners:
                    x, y = corner.ravel()
                    explosionPointsX.append(x)
                    explosionPointsY.append(y)
#                    cv2.circle(mask, (x,y), 4, 255, -1)
                
            if 'explosionPointsX' in locals():
                X = int(np.array(explosionPointsX).mean())
                Y = int(np.array(explosionPointsY).mean())
#                cv2.circle(mask, (X,Y), 3, 255, -1)
            
            if time()-entryTime >= 6 or (counterCurrent == counterOld and counterCurrent != 0):
#                cv2.destroyAllWindows()
                if 'X' in locals():
                    return (X, Y)
                else:
                    return (False, False)
            
#            cv2.imshow("mask", mask)
            counterOld = counterCurrent
#            if cv2.waitKey(1) == 27:
#                break
    
        except Exception as e:
            print(str(e))
            continue
#    cv2.destroyAllWindows()


def divide_image(img, divisions):
    height, width, _ = img.shape
    L = []
    for i in range(divisions):
        L.append(img[:, int(i/divisions*width):int((i+1)/divisions*width)])
    return L

def get_max_height(img, idx):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)
    if corners is not None:
        corners = np.int0(corners)
        xd, yd = 1000, 1000
        for c in corners:
            x, y = c[0][0], c[0][1]
            if y < yd:
                yd = y
                xd = x
        return xd, yd
    else:
        return int((idx+0.5)*img.shape[1]), 0
        
def detectWorldHighestPoints(divisions):
    #READING IMAGE FROM GAME
    img = np.array(app["Pocket Tanks Deluxe"].CaptureAsImage())
    
    #CROPPING  AND CONVERTING IMAGES
    img = img[31+33:rows, 10:cols]    #31 for the titlebar and 33 for the names
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_world, upper_world)
    resWorld = cv2.bitwise_and(img, img, mask = mask)
    images = divide_image(resWorld, divisions)
    points = []
    for i in range(len(images)):
        x, y = get_max_height(images[i], i)
        points.append(y)
    return points


def detectTankCooridnatesByFeatureMatching():
    #READING IMAGE FROM GAME
    img = np.array(app["Pocket Tanks Deluxe"].CaptureAsImage())
    
    #CROPPING AND CONVERTING IMAGES
    img = img[31+33:rows, 10:cols]    #31 for the titlebar and 33 for the names
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_tank1, upper_tank1)
    resYellowTank = cv2.bitwise_and(img, img, mask = mask)
    
    mask = cv2.inRange(hsv, lower_tank2, upper_tank2)
    resBlueTank = cv2.bitwise_and(img, img, mask = mask)
#    while True:
#        cv2.imshow("rbT", resBlueTank)
#        cv2.imshow("ybT", resYellowTank)
#        
#        if cv2.waitKey(1) == 27:
#            cv2.destroyAllWindows()
    
    kpResYellowTank, desResYellowTank = orb.detectAndCompute(resYellowTank, None)
    yellowTankMatches = bf.match(desYellowTank, desResYellowTank)
    yellowTankMatches = sorted(yellowTankMatches, key=lambda x:x.distance)
    
    kpResBlueTank, desResBlueTank = orb.detectAndCompute(resBlueTank, None)
    blueTankMatches = bf.match(desBlueTank, desResBlueTank)
    blueTankMatches = sorted(blueTankMatches, key=lambda x:x.distance)
    
    blueTankPointsX = []
    blueTankPointsY = []
    yellowTankPointsX = []
    yellowTankPointsY = []
    
    for i in range(10):
        blueTankPointsX.append(kpResBlueTank[blueTankMatches[i].trainIdx].pt[0])
        yellowTankPointsX.append(kpResYellowTank[yellowTankMatches[i].trainIdx].pt[0])
        blueTankPointsY.append(kpResBlueTank[blueTankMatches[i].trainIdx].pt[1])
        yellowTankPointsY.append(kpResYellowTank[yellowTankMatches[i].trainIdx].pt[1])
        
    return (int(np.mean(yellowTankPointsX)), int(np.mean(yellowTankPointsY))), (int(np.mean(blueTankPointsX)), int(np.mean(blueTankPointsY)))


def detectTankCooridnatesByTemplateMatching():
    press(("a"))
    #READING IMAGE FROM GAME
    img = np.array(app["Pocket Tanks Deluxe"].CaptureAsImage())
    
    #CROPPING AND CONVERTING IMAGES
    img = img[31+33:rows, 10:cols]    #31 for the titlebar and 33 for the names
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_tank1, upper_tank1)
    resYellowTank = cv2.bitwise_and(img, img, mask = mask)
    resYellowTank = cv2.cvtColor(resYellowTank, cv2.COLOR_HSV2BGR)
    resYellowTank = cv2.cvtColor(resYellowTank, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(resYellowTank,yellowTankTemplate2,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
        
    while True:
        threshold -= 0.05
        loc = np.where( res >= threshold)

        if len(loc[0]) != 0:
            yellowTankPoint = (loc[1][0], loc[0][0])
            break
            
    mask = cv2.inRange(hsv, lower_tank2, upper_tank2)
    resBlueTank = cv2.bitwise_and(img, img, mask = mask)
    resBlueTank = cv2.cvtColor(resBlueTank, cv2.COLOR_HSV2BGR)
    resBlueTank = cv2.cvtColor(resBlueTank, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(resBlueTank, blueTankTemplate2, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
        
    while True:
        threshold -= 0.05
        loc = np.where( res >= threshold)

        if len(loc[0]) != 0:
            blueTankPoint = (loc[1][0], loc[0][0])
            break
    
    return yellowTankPoint, blueTankPoint

###############################################################################
#Working with the keyboard
###############################################################################
VK_CODE = {
           'spacebar':0x20,
           'left_arrow':0x25,
           'up_arrow':0x26,
           'right_arrow':0x27,
           'down_arrow':0x28,
           'a':0x41
}

def adjustPowerAndAngle(predictedPower, predictedAngle):
        sleep(1/timeSpeedFactor)
        
        global power, angle
        
        if predictedPower > power:
            for i in range(predictedPower - power):
                press(('up_arrow'))
        elif predictedPower < power:
            for i in range(power - predictedPower):
                press(('down_arrow'))
        
        if predictedAngle > angle:
            for i in range(predictedAngle - angle):
                press(('left_arrow'))
        elif predictedAngle < angle:
            for i in range(angle - predictedAngle):
                press(('right_arrow'))
                
        power, angle = predictedPower, predictedAngle
    
def press(*args):   
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        sleep(0.05)
        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
        sleep(0.05)

###############################################################################
#Process connection
###############################################################################
app = application.Application().connect(path='pockettanks.exe')
###############################################################################
#Initializing global variables
###############################################################################

rows, cols = 639, 816
rows-=10+146       #10 for the bottom border and 146 for the GUI
cols-=10

timeSpeedFactor = float(input("Enter the speed hack factor: "))
power = int(input("Enter the current in-game power: "))
angle = int(input("Enter the current in-game angle: "))

###############################################################################
#Neural network functions
###############################################################################

def calculateDistance():
    return sqrt((blueTankCoordinates[0]-collisionPoint[0])**2+(blueTankCoordinates[1]-collisionPoint[1])**2)


def predict(state):
    output = model.predict(state.reshape(-1, 25))
    power, angle = output[0]
#    print("predicted power:", power)
#    print("predicted angle:", angle)
    power = int(np.clip(power, 5, 100))
    angle = int(np.clip(angle, 5, 85))
    print("predicted power:", power)
    print("predicted angle:", angle)
    return power, angle

###############################################################################
#Model creation/loading
###############################################################################
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
#model = Sequential()
#model.add(Dense(units=32, activation='relu', input_dim=10))
#model.add(Dense(units=32, activation='relu'))
#model.add(Dense(2))
#model.compile(loss=_loss_tensor, optimizer=sgd)
#
#import keras.losses
#keras.losses.custom_loss = custom_loss
    
model = load_model(filepath='.\model.h5')

###############################################################################
#Program main loop
###############################################################################

#Waiting for the user to switch to the game
print("Okay now switch to the game, you have two seconds.")
sleep(2)
#dX = np.random.randint(2, 10)
dX = 2
while True:
    try:
        yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByFeatureMatching()
    except Exception as e:
        yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByTemplateMatching()
          
    worldHighestPoints = detectWorldHighestPoints(20)
    worldLowestAngle = detectWorldLowestAngle(yellowTankCoordinates, blueTankCoordinates)
    
    dX = np.random.randint(0, 5)
    dY = np.random.randint(0, 1)
#    state = np.array([blueTankCoordinates[0] - yellowTankCoordinates[0], blueTankCoordinates[1] - yellowTankCoordinates[1], \
#                    worldHighestPoints[0][0], worldHighestPoints[0][1], worldHighestPoints[1][0], \
#                    worldHighestPoints[1][1], worldHighestPoints[2][0], worldHighestPoints[2][1], \
#                    d])
    state = np.array([blueTankCoordinates[0] - yellowTankCoordinates[0], yellowTankCoordinates[1] - blueTankCoordinates[1], \
                    *worldHighestPoints, worldLowestAngle, dX, dY])

    #make the AI make a decision
    predictedPower, predictedAngle = predict(state)
    #change power/angle
    adjustPowerAndAngle(predictedPower, predictedAngle)
    #FIRE and wait for the collision and calculate the distance between the projectile and the blue tank
    press(('spacebar'))
    collisionPoint = detectCollisionCoordinates()
    distance = calculateDistance()
    print("distanceX:", int(collisionPoint[0]-blueTankCoordinates[0]), "  desired distanceX:", dX)
#    print("distanceY:", int(collisionPoint[1]-blueTankCoordinates[1]), "\tdesired distanceY:", dY)
#        
#    if collisionPoint[0]-blueTankCoordinates[0]>40:
#        dX += 8
#    if collisionPoint[0]-blueTankCoordinates[0]<-40:
#        dX -= 8
    #Wait for the world crumblings
    sleep(3/timeSpeedFactor)
    #wanna make a move?
    sleep(4/timeSpeedFactor)
    #Make the blue tank fire away 
    collisionPoint = detectCollisionCoordinates()
    
    sleep(2/timeSpeedFactor)
    
    
