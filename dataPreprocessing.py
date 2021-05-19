import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import avg,readDS,percent_increase
plt.style.use('ggplot')
from scipy.interpolate import interp1d
from scipy.interpolate import splrep,splev
import warnings
warnings.filterwarnings("ignore")


class Interpolations:
	def __init__(self,df,col,plot):
		self.plot = plot
		self.col = col
		temp_df = df.dropna(subset = [col])
		self.t = temp_df[col].values.reshape(-1)
		self.part_time = (pd.to_datetime(temp_df["Timestamp"]).values.astype(int)/ 10**9  ).reshape(-1)      
		self.all_time = (pd.to_datetime(df["Timestamp"]).values.astype(int)/ 10**9  ).reshape(-1)  
		self.onlyna_time = (pd.to_datetime(df["Timestamp"][pd.isna(df["Open"])]).values.astype(int)/ 10**9  ).reshape(-1) 
		

	def OneDInterpolation(self):
		f = interp1d(self.part_time,self.t,kind = "cubic")
		if self.plot:
		    fig = plt.figure(figsize=(14,14))
		    plt.plot(self.part_time,self.t, self.onlyna_time ,f(self.onlyna_time),"r.",ms = 2 )           
		    plt.title(self.col,size = 20 )   
		    plt.legend(["real","imputed"])
		    plt.show()
		return f(self.all_time)
	
	
	def SplineInterpolation(self):
		f = splrep(self.part_time,self.t)
		if self.plot:
			fig = plt.figure(figsize=(14,14))
			plt.plot(self.part_time,self.t, self.onlyna_time, splev(self.onlyna_time,f),"r.",ms = 2 )
			plt.title(self.col,size = 20)
			plt.legend(["real","imputed"])              
			plt.show()
		return splev(self.all_time,f)
		
	def AVG_LOCF_NOCB(self):
		f1 = interp1d(self.part_time,self.t,kind = "previous")
		f2 = interp1d(self.part_time,self.t,kind = "next")
		if self.plot:
			fig = plt.figure(figsize=(14,14))
			plt.plot(self.part_time,self.t,  self.onlyna_time,avg(f1(self.onlyna_time),f2(self.onlyna_time)),"r.",ms = 2 )
			plt.title(self.col,size = 20 )
			plt.legend(["real","imputed"])           
			plt.show()
		return avg(f1(self.all_time),f2(self.all_time))


def MV_imputation(spline,plot,norm):
	df = readDS(time=True)
	inter_VB = Interpolations(df,"Volume_(BTC)",plot)
	inter_VC = Interpolations(df,"Volume_(Currency)",plot)
	inter_O = Interpolations(df,"Open",plot)
	inter_C = Interpolations(df,"Close",plot)
	inter_H = Interpolations(df,"High",plot)
	inter_L = Interpolations(df,"Low",plot)
	inter_WP = Interpolations(df,"Weighted_Price",plot)
	
	df["Volume_(BTC)"] = inter_VB.AVG_LOCF_NOCB() 
	df["Volume_(Currency)"] = inter_VC.AVG_LOCF_NOCB() 
	
	if spline:
		df["Open"] = inter_O.SplineInterpolation() 
		df["Close"] = inter_C.SplineInterpolation() 		
		df["High"] = inter_H.SplineInterpolation()	
		df["Low"] = inter_L.SplineInterpolation() 
		df["Weighted_Price"] = inter_WP.SplineInterpolation() 
	else:
		df["Open"] = inter_O.OneDInterpolation() 
		df["Close"] = inter_C.OneDInterpolation() 		
		df["High"] = inter_H.OneDInterpolation()	
		df["Low"] = inter_L.OneDInterpolation() 
		df["Weighted_Price"] = inter_WP.OneDInterpolation() 
	
	df["change in price"] = percent_increase(df["Weighted_Price"],norm)
	df.to_csv("Datasets/final_Bitcoin_dataset.csv")


    
def main(spline,plot,norm):
	MV_imputation(spline,plot,norm)

	
if __name__ == '__main__':
    main()

