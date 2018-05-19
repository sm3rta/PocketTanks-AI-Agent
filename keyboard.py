# -*- coding: utf-8 -*-
"""
Created on Sat May 19 23:05:17 2018

@author: Smarta
"""
from Variables import *
import win32api, win32con
from time import time, sleep
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

