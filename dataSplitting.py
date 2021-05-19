import pandas as pd
import numpy as np
import os
from utils import shift
import argparse
from sklearn.model_selection import train_test_split


	
def Split(train_size,price_only,shift_param,per_change, features = ["Open","High","Low","Close"],save = True):
	df = pd.read_csv("Datasets/final_Bitcoin_dataset.csv",index_col = 0)
	if per_change:
		gls = "change in price"
	else:
		gls = "Weighted_Price"
	if price_only:
		X = df["Weighted_Price"][:-1]
	else:
		X = df[features][:-1]
	y = df[gls][:-1]
	X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                shuffle = False,
                                                  train_size = train_size)
	try : 
		os.mkdir("Datasets/split")
	except:
		1

	X_train,y_train = shift(X_train,y_train,shift_param)
	X_test,y_test = shift(X_test,y_test,shift_param)
			
	if save:
		np.save("./Datasets/split/X_train.npy",X_train)
		np.save("./Datasets/split/X_test.npy",X_test)
		np.save("./Datasets/split/y_train.npy",y_train)
		np.save("./Datasets/split/y_test.npy",y_test)
	else:
		return X_train,y_train



def main(train_size,price_only,shift_param,per_change):

	Split(train_size,price_only,shift_param,per_change)

	
if __name__ == '__main__':
    main()
