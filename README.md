# PocketTanks-AI-Agent
An AI for Pocket Tanks
## How to use :
1. Install necessary dependencies, in requirements.txt.
1. Open PocketTanks.exe.
1. Make sure the only weapon that can be used is "Single Shot" in options.
1. Enter target practice or 1 player mode.
1. Make sure that your tank is cyan and the other one is orange.
1. Run the script "PocketTanksAgent.py"
1. You'll be asked to enter the values of 3 things
   * Speed hack factor, I used Cheat Engine to make the game faster, particularly useful when gathering data from the game. If you use speed hack as well, it's necessary to enter it in the program at the beginning to synchronize the timing between the program and the game. If you don't use it, just enter 1.
   * The angle currently displayed in the game, that will be updated by the program.
   * The in game power.
1. Press Enter and switch to the game, the agent should start playing.


## File breakdown :
File | Discription
--- | ---
OpenCV.py | Contains OpenCV based functions which detect tanks coordinates, projectile coordinates and world shape features
PocketTanksAgent.py | Starts the agent
PocketTanksAgent.sln | Visual Studio Solution
Variables.py | Contains global variables used by both PocketTanksAgent.py and DataGathering.py
data.csv	| Contains data I collected from the game 
dataGathering.py | Generates random values and collects results, updates data.csv
keyboard.py	| Contains keyboard related functions, responsible for updating ingame power and angle
model.h5	| keras sequential model
requirements.txt	| Contains project dependencies, generated by Visual Studio, although I'm sure not all of them are necessary
tank_blue.png	| Blue tank template for feature matching
tank_blue3.png	| Blue tank template for template matching
tank_yellow.png	| Yellow tank template for feature matching
tank_yellow3.png	| Yellow tank template for template matching
training.py | Creates a neural network, trains it and saves the model to model.h5


## Model Features :
Name | Discription
--- | ---
deltaX and deltaY | horizontal and vertical distances between the two tanks
w0 to w19 | Highest point y value in 20 divisions of the screen
worldLowestAngle | The highest angle of all the angles between the two tanks and every co-ordinate of the world's highest points ![alt text](https://github.com/sm3rta/PocketTanks-AI-Agent/blob/master/worldLowestAngleDemo.png "worldLowestAngleDemo")
distanceX and distanceX | horizontal and vertical distances between the projectile when it hit the ground (or the tank) and the other tank
power and angle | The power and angle used in this situation that resulted in distanceX and distanceY
