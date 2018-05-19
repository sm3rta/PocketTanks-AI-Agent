from pywinauto import application

#Process connection
app = application.Application().connect(path='pockettanks.exe')


#Initializing global variables
rows, cols = 639, 816
rows-=10 + 146       #10 for the bottom border and 146 for the GUI
cols-=10

timeSpeedFactor = float(input("Enter the speed hack factor: "))
power = int(input("Enter the current in-game power: "))
angle = int(input("Enter the current in-game angle: "))
