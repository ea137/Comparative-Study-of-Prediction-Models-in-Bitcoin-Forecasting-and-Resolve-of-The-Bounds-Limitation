import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import argparse
from sklearn import linear_model,svm
from utils import readSplit, predict_price,scale_svm
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


########################## models ########################
class models:
	def __init__(self,model_type):
		self.name = model_type
		self.X_train, self.y_train,self.X_test,self.y_test = readSplit()
		if self.name == "lr":
			self.run = self.LR()
		elif self.name == "rf":
			self.run = self.RF()
		elif self.name == "dtr":
			self.run = self.DTR()
		elif self.name == "xgboost":
			self.run = self.XGBOOST()
		elif self.name == "svr":
			self.training_max = 19663.30
			self.training_min = 752
			self.run = self.SVR()
		else:
			raise Exception("unknown model " + str(self.name) + ", choose one of the following [xgboost,lr,dtr,rf,svr] ")
	def RF(self):
	
		rf_reg = RandomForestRegressor(criterion='mse',
                               min_samples_leaf=1,
                                min_samples_split=7,
                               max_depth=15,
                               n_estimators=50,ccp_alpha=0.0)
		rf_reg.fit(self.X_train,self.y_train)
		return rf_reg
		
	def DTR(self):
	
		dtr = DecisionTreeRegressor(max_depth=18,
                           criterion = 'mse',
                           max_leaf_nodes=50)
		dtr.fit(self.X_train,self.y_train)
		return dtr
		
	def LR(self):

		lr = linear_model.LinearRegression()
		lr.fit(self.X_train,self.y_train)
		return lr
		
	def XGBOOST(self):
	
		boost =  xgb.XGBRegressor(objective ='reg:linear', 
								min_child_weight=10,
								booster='gbtree',
								learning_rate = 0.1,
		            			max_depth = 7,
		            			alpha = 10, n_estimators = 100)

		boost.fit(self.X_train, self.y_train)
		return boost
	
	def SVR(self ):
		scaledX,scaledy = scale_svm(self.X_train,self.y_train,
								self.training_max, self.training_min)
		svr = svm.SVR(kernel='rbf', C=0.75)
		svr.fit(scaledX,scaledy )
		return svr
		

def main(model_type,metric,predict,save,plot,train_size,norm,per_change):

	if plot or save:
		predict = True
	print("Started Training...")
	model = models(model_type)
	print("Training Done!")
	if predict : 
		pred = predict_price(model, plot,metric,norm,per_change)
	if save:
		try : 
			os.mkdir("results")
		except:
			1
		filename = "./results/predictions_"+model.name+"_"+str(int(train_size*100))+".npy"
		np.save(filename,pred)
	return model
if __name__ == '__main__':
    main()
