#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/31 11:36
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""

# Part 1 - Data Preprocessing Template
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
# Importing the dataset
os.chdir(r"C:\Users\Administrator\Desktop\wuquan")
dataset = pd.read_csv('test-03.csv')
X = dataset.iloc[:, :3].values
y = dataset.iloc[:, 3:6].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units=5, kernel_initializer='RandomNormal', activation='tanh', input_dim=3))
# Adding the output layer
classifier.add(Dense(units=3, kernel_initializer='RandomNormal', activation='linear'))
# Compiling the ANN
sgd = optimizers.SGD(lr=0.03, momentum=0.9, decay=1e-5, nesterov=True)
classifier.compile(optimizer='sgd', loss='mse', metrics=['mae'])
# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size=20, epochs=2000, validation_split=0.2)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Calculating R2
from sklearn.metrics import r2_score

rr = r2_score(y_test, y_pred)
# Listing all data in history
print(history.history.keys())
# Summarizing history for mae
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['val_loss'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train mae', 'train loss', 'val mae', 'val loss'], loc='upper right')
plt.show()

plt.scatter(X_test[:, :1], y_test[:, :1], color='red')
plt.scatter(X_test[:, :1], y_pred[:, :1], color='blue')
plt.title('PAO vs YDND')
plt.xlabel('PAO')
plt.ylabel('YDND')
plt.show()

# store model
os.chdir(r"C:\Users\Administrator\Desktop\wuquan")
pd.to_pickle(classifier, r"AnnClassifier")
pd.to_pickle(sc, r"MinMaxScaler")