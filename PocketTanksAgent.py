import numpy as np

from keras.models import load_model

from OpenCV import *

#Neural network functions
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

#Model loading
model = load_model(filepath='.\model.h5')


#Waiting for the user to switch to the game
print("Okay now switch to the game, you have two seconds.")
sleep(2)

#Program main loop
while True:
    try:
        yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByFeatureMatching()
    except Exception as e:
        yellowTankCoordinates, blueTankCoordinates = detectTankCooridnatesByTemplateMatching()

    worldHighestPoints = detectWorldHighestPoints(20)
    worldLowestAngle = detectWorldLowestAngle(yellowTankCoordinates, blueTankCoordinates)

    dX = np.random.randint(0, 5)
    dY = np.random.randint(0, 1)

    state = np.array([blueTankCoordinates[0] - yellowTankCoordinates[0],\
                      yellowTankCoordinates[1] - blueTankCoordinates[1], \
                    *worldHighestPoints, worldLowestAngle, dX, dY])

    #make the AI make a decision
    predictedPower, predictedAngle = predict(state)
    #change power/angle
    adjustPowerAndAngle(predictedPower, predictedAngle)
    #FIRE and wait for the collision and calculate the distance between the
    #projectile and the blue tank
    press(('spacebar'))
    collisionPoint = detectCollisionCoordinates()

    print("distanceX:", int(collisionPoint[0] - blueTankCoordinates[0]), "  desired distanceX:", dX)

    #Wait for the world crumblings
    sleep(3 / timeSpeedFactor)
    #wanna make a move?
    sleep(4 / timeSpeedFactor)
    #wait for the blue tank fire away
    press(('spacebar'))
    #replace with collisionPoint = detectCollisionCoordinates()
    #if in player vs com mode, to wait for the projectile
    #Wait for the world crumblings
    sleep(3 / timeSpeedFactor)
