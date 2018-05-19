###############################################################################
#Imports
###############################################################################
import pandas as pd
import cv2
import numpy as np

from time import time, sleep
import win32api
import win32con
from math import sqrt

from keras.models import Sequential, load_model
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from keras import optimizers

from OpenCV import *

#Loading the dataframe
names = ["deltaX", "deltaY", \
         "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", \
         "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "worldLowestAngle", \
         "distanceX", "distanceY",\
         "power","angle"]

df = pd.read_csv('./data.csv', index_col = None, names = names)

samples = len(df[df.columns[0]])

#Waiting for the user to switch to the game
print("Okay now switch to the game, you have two seconds.")
sleep(2)

i = 0
#Program main loop
while i < np.inf:
#for randomAngle in range(65, 80, 1):
#    for randomPower in range(75, 90, 2):
    
        randomPower = np.random.randint(40, 60)
        randomAngle = np.random.randint(60, 70)
#        randomAngle = 70
        try:
            yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByFeatureMatching()
        except Exception as e:
            yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByTemplateMatching()
    
        worldHighestPoints = detectWorldHighestPoints(20)
        worldLowestAngle = detectWorldLowestAngle(yellowTankCoordinates, blueTankCoordinates)
            
        print("random power:", randomPower)
        print("random angle:", randomAngle)
        
        adjustPowerAndAngle(randomPower, randomAngle)
        
        #FIRE and wait for the collision and calculate the distance between the
        #projectile and the blue tank
        collisionPoint = detectCollisionCoordinates()
        if collisionPoint[0]:
            distanceX = blueTankCoordinates[0] - collisionPoint[0]
            distanceY = blueTankCoordinates[1] - collisionPoint[1]
            print("distance:", distanceX)
    
            df.loc[samples] = [blueTankCoordinates[0] - yellowTankCoordinates[0], yellowTankCoordinates[1] - blueTankCoordinates[1], \
                   *worldHighestPoints, worldLowestAngle, \
                   distanceX, distanceY, \
                   power, angle]
    
            samples += 1
    
            #Wait for the world crumblings
            sleep(3 / timeSpeedFactor)
            #Make the blue tank fire away
#            sleep(1/timeSpeedFactor)
            press(('spacebar')) 
            sleep(1 / timeSpeedFactor)
            

        else:
            press(('spacebar'))
            sleep(1 / timeSpeedFactor)
            
    
        #save every 20 samples, just in case...
        if i % 10 == 0:
            df.to_csv('data.csv', sep=',', index=False, header=None)
        i+=1


###############################################################################
#CSV saving, must be done manually after ctrl+c'ing the program since the
#infinite loop
###############################################################################
df.to_csv('data.csv', sep=',', index=False, header=None)
