import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf
df = pd.read_csv("diabetes.csv")
x,y = df.drop(["Outcome"], axis=1), df["Outcome"] 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))


model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')
y_train = np.asarray(y_train)
model.fit(x_train,y_train,epochs=250)
preg = input("Pregnancies: ")
glucose = input("glucose: ")
blood_press = input("blood pressure: ")
skin_thickness = input("skin thickness: ")
insulin = input("Insulin: ")
BMI = input("BMI: ")
DiabetesPedigreeFunction = input("DiabetesPedigreeFunction: ")

your_features = [[preg,glucose,blood_press,skin_thickness,insulin,BMI,DiabetesPedigreeFunction]]
prediction = model.predict(your_features)
print(prediction)
