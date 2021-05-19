import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import argparse
from utils import readSplit, predict_price
from sklearn.metrics import mean_absolute_error as MSE
import math , scipy
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU

"""
from math import sqrt
"""



########################## models ########################
class models:
	def __init__(self,model_type):
		self.name = model_type
		self.X_train, self.y_train,self.X_test,self.y_test = readSplit()

		if self.name == "lstm":
			self.run = self.LSTM()
		elif self.name == "gru":
			self.run = self.GRU()
		else:
			raise ValueError("unknown model " + str(self.name) + ", choose one of the following [lstm,gru] ")


	def LSTM(self):
		self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
		self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

		lstm = Sequential()
		lstm.add(LSTM(1, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
		lstm.add(Dense(1))
		lstm.compile(loss='mean_squared_error', optimizer='adam')

		#history = lstm.fit(self.X_train, self.y_train, epochs=10, batch_size=3, validation_data=(self.X_test, self.y_test), verbose=True, shuffle=False)
		#lstm.save("./lstm_weights.h5")
		lstm.load_weights("./lstm_weights_updatedMethod.h5")
		return lstm

	def GRU(self):
		self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
		self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
				
		gru = Sequential()
		gru.add(GRU(1, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
		gru.add(Dense(1))
		gru.compile(loss='mean_squared_error', optimizer='adam')
		#history = gru.fit(self.X_train, self.y_train, epochs=5, batch_size=3, validation_data=(self.X_test, self.y_test), verbose=True,shuffle=False)
		#gru.save("./gru_weights.h5")
		gru.load_weights("./gru_weights_updatedMethod.h5")
		return gru


def main(model_type,metric,predict,save,plot,train_size,norm,per_change):

	if plot or save:
		predict = True
	model = models(model_type)
	if predict :
		pred = predict_price(model, plot,metric,norm,per_change)
	if save:
		try :
			os.mkdir("results")
		except:
			1
		filename = "./results/predictions_"+model.name+"_"+str(int(train_size*100))+".npy"
		np.save(filename,pred)


if __name__ == '__main__':
    main()
