import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv("diabetes.csv")
x,y = df.drop(["Outcome"], axis=1), df["Outcome"] 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
model = Sequential()

model.add(Dense(9,activation='relu'))
model.add(Dense(9,activation='relu'))


model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy')

y_test = np.asarray(y_test)
y_train = np.asarray(y_train)
model.fit(x_train,y_train,epochs=250)
earlystop = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=25)

model.fit(x_train,y_train,epochs=99999999,validation_data=(x_test,y_test),callbacks=[earlystop])


new_prediction = [[0,40,12,3,800,22,0.2,18]]

your_features = [[preg,glucose,blood_press,skin_thickness,insulin,BMI,DiabetesPedigreeFunction,Age]]
prediction = model.predict(new_gem)
print(prediction)
