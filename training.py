import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

#MODEL creation
ADAM = optimizers.Adam(lr=0.0005)
MODEL = Sequential()
MODEL.add(Dense(units=32, activation='relu', input_dim=25))
MODEL.add(Dense(units=64, activation='relu'))
#MODEL.add(Dense(units=64, activation='relu'))
MODEL.add(Dense(2))
MODEL.compile(loss='mean_squared_error', optimizer=ADAM)

#Loading the dataframe
NAMES = ["deltaX", "deltaY", \
         "w0", "w1", "w2", "w3", "w4", "w5", \
         "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14"\
         , "w15", "w16", "w17", "w18", "w19", "worldLowestAngle"\
         , "distanceX", "distanceY", "power", "angle"]
DF = pd.read_csv('data.csv', index_col=None, names=NAMES)

#Cleaning data
DF = DF[DF.distanceX < 70]
DF = DF[DF.distanceX >= -70]
DF = DF[DF.angle > 65]
DF = DF[DF.angle < 75]
#DF = DF[DF.power > 50]
#DF = DF[DF.power < 85]
X = DF.iloc[:, :-2]
Y = DF.iloc[:, -2:]

SAMPLES = len(DF[DF.columns[0]])
MODEL.summary()

MODEL.fit(x=np.array(X).reshape(-1, 25), y=np.array(Y).reshape(-1, 2), \
          batch_size=SAMPLES, epochs=9000, verbose=1)

MODEL.save('MODEL.h5')
