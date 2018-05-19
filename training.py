###############################################################################
#Imports
###############################################################################
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras_contrib.layers.advanced_activations import PELU
from keras import optimizers
###############################################################################
#Model creation/loading
###############################################################################
Adam = optimizers.Adam(lr=0.0005)
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=25))
#model.add(PELU())
model.add(Dense(units=64, activation='relu'))
#model.add(Dense(units=128, activation='relu'))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer="Adam")
###############################################################################
#Loading the dataframe
###############################################################################

#names = ["deltaX", "deltaY","WorldHighest1X","WorldHighest1Y","WorldHighest2X","WorldHighest2Y","WorldHighest3X","WorldHighest3Y","Distance","power","angle"]
#names = ["deltaX", "deltaY", "worldLowestAngle", "Distance","power","angle"]
names = ["deltaX", "deltaY", \
         "w0", "w1", "w2", "w3", "w4", "w5", \
         "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14"\
         , "w15", "w16", "w17", "w18", "w19","worldLowestAngle"\
         , "distanceX", "distanceY","power","angle"]
df = pd.read_csv('data.csv', index_col = None, names = names)
#Cleaning data
df = df[df.distanceX < 70]
df = df[df.distanceX >= -70]
df = df[df.angle > 65]
df = df[df.angle < 75]
#df = df[df.power > 50]
#df = df[df.power < 85]
#input and output
x = df.iloc[:, :-2]
y = df.iloc[:, -2:]

samples = len(df[df.columns[0]])
model.summary()
#model.fit(x=np.array(x).reshape(-1, 9), y = np.array(y).reshape(-1, 2), epochs=12000, batch_size=samples, verbose=1)
model.fit(x=np.array(x).reshape(-1, 25), y = np.array(y).reshape(-1, 2), batch_size=samples, epochs=9000, verbose=1)

model.save('model.h5')

#48,48,48
#29.6