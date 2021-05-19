import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt


def percent_increase(col,norm):
    """
    in the dataset, the prices are recorded each minute so the changes are small.
    If the changes are higher (i.e the interval of recording was larger), use a smaller norm
    """
    col = col.values
    res = np.ones(col.shape)
    res[0] = res[0]* -1
    col_decal = col[1:]
    col = col[:-1]
    perc = ((col_decal - col) / (col + col_decal)) * norm
    res[1:] = res[1:] * perc
    return res
    
def percent_convertion(X,y_pred,norm):

	res = np.ones(y_pred.shape)
	for i,xy in enumerate(zip(X,y_pred)):
		#print(xy)
		res[i] = (xy[0].reshape(-1)[-1] * (norm - xy[1] )) / (norm + xy[1])
	return res


def avg(arr1,arr2):
    return (arr1+arr2)/2

def scale_svm(X,y,maxi,mini):
	X = (X -mini) / (maxi - mini)
	y = (y -mini) / (maxi - mini)
	X[X<0] = 0
	y[y<0] = 0
	X[X>1] = 1
	y[y>1] = 1
	return X,y

def readDS(time=True):
	df = pd.read_csv("Datasets/bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv")
	df = df.dropna(subset=["Timestamp"])
	if time:
		df["Timestamp"] = pd.to_datetime(df['Timestamp'],unit='s')
		df = df[df["Timestamp"] > "2017"] # only keeping data post 2017
	return df

def readfinDS():
	df = pd.read_csv("Datasets/final_Bitcoin_dataset.csv",index_col = 0)
	return df
	
def readSplit():

	X_train = np.load("./Datasets/split/X_train.npy")
	X_test = np.load("./Datasets/split/X_test.npy")
	y_train = np.load("./Datasets/split/y_train.npy")
	y_test = np.load("./Datasets/split/y_test.npy")
	
	return X_train, y_train, X_test, y_test
	
def correlation():
	# needs to first run dataCleaning.py to be able to display the correlation, alternatively, run main.py then run correlation()
	df = readfinDS()
	gls = df.columns.to_list()[-1]
	features = df.columns.to_list()[1:-1]
	X = df[features][:-1]
	y = df[gls][:-1]
	X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                shuffle = False,
                                                  train_size = .7)
	X_train, y_train = shift(X_train,y_train)
	X_test, y_test = shift(X_test,y_test)
	corr = pd.DataFrame(np.c_[X_train,y_train]).corr(method='pearson').round(2)
	plt.figure(figsize=(14,14))
	plt.xticks(size =20)
	plt.yticks(size =20)
	mask = np.ones((7,7))
	mask[:,6] = 0
	sns.heatmap(corr,square = True,
		       	cmap='RdGy',
		        mask=  mask,
				yticklabels=["Open", "High", "Low", "Close", "volume(BTC)", "volume(Currency)", "Weighted Price"],
				xticklabels=["Open", "High", "Low", "Close", "volume(BTC)", "volume(Currency)", "Weighted Price"])
	plt.show()

def shift(X,y,shift_param):
	y = y.to_numpy()
	X = X.to_numpy()
	reshape = False
	if shift_param ==1:
		y_new = y[shift_param:] 
		X_new = X[:-shift_param]
	else:
		y_new = y[shift_param:]
		X_new = []
		if len(X.shape) == 1:
			temp = np.zeros(shift_param)
		else:

			if X.shape[1] == 1:
			 	temp = np.zeros(shift_param)
			else:
				temp = np.zeros((shift_param,X.shape[-1]))
				reshape = True

				
		for i in range(shift_param):
			temp[i] = X[i]
		for i in X[shift_param:]:
			X_new.append(np.array(temp))
			temp = np.roll(temp,shift_param-1)
			temp[-1] = i
		X_new = np.array(X_new)
	if reshape:
		X_new = X_new.reshape(-1,shift_param*X.shape[1])
	else:
		try:
			X_new = X_new.reshape(-1,X.shape[1]*shift_param)
		except:
			X_new = X_new.reshape(-1,1*shift_param)
	#print(X.shape,len(X.shape),reshape)
	return X_new,y_new

def metrics_(y_test,pred,name,metric):
	if metric == "mse":
		res = MSE(y_test,pred)
	elif metric == "mae":
		res = MAE(y_test,pred)
	elif metric == "r2":
		res = R2(y_test,pred)
	elif metric == "all":
		print("MSE of " + name + " : " + str(round(MSE(y_test,pred),2)) )
		print("MAE of " + name + " : " + str(round(MAE(y_test,pred),2)) )
		print("R2 of " + name + " : " + str(round(R2(y_test,pred),2)) )
		return True
	else:
		raise Exception("unknown model " + str(name) + ", choose one of the following [mse,mae,r2] ")
		
	print(metric.upper() + " of " + name + " : " + str(round(res,2)) )
	return True
		
def predict_price(model, plot, metric,norm,per_change):
	if model.name == "svr" and per_change == False:
		X,y = scale_svm(model.X_test,model.y_test,model.training_max,model.training_min)
		pred = model.run.predict(X)
		pred = pred * (model.training_max-model.training_min) + model.training_min
		y_true = model.y_test
		#return
	else:
		pred = model.run.predict(model.X_test)
		y_true = model.y_test
	if per_change:
		pred = percent_convertion(model.X_test,pred,norm)
		y_true = percent_convertion(model.X_test,model.y_test,norm)
	res = metrics_(y_true,pred,model.name,metric)
	
	if plot:
		plt.figure(figsize=(14,14))
		plt.plot(pred,"--o",label = "prediction")
		plt.plot(y_true,"-",label = "real price")

		plt.xlabel("Time",size = 20)
		plt.ylabel("Weighted Price",size = 20)
		plt.xticks([])
		plt.legend(fontsize = 20)
		plt.title(model.name.upper() + " predictions" ,size = 20)	
		plt.show()
	return pred


