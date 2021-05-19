import argparse
import dataSplitting, dataPreprocessing
from utils import predict_price

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args():
	parser = argparse.ArgumentParser(description="All options", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--spline", type=boolean_string, default=True,
                        help='True -> Imputes missing values using spline interpolation False -> Imputes missing values using one dimensional interpolation')
	parser.add_argument("--plot_imputation", type=boolean_string, default=False, help="True -> Plots the imputation results")                  
	parser.add_argument("--train_size", type=float, default=.3,help = "Size of training data (in percents)")
	parser.add_argument("--price_only", type=boolean_string, default=True, help = "True -> Uses only price as a feature, False -> Uses 4 features [Open, High, Low, Close]")
	parser.add_argument("--shift_param", type=int, default=1, help = "Number of past features to consider")  	             
	parser.add_argument("--model_type", type=str, default="lr",help = "Availabe options : [xgboost,lr,dtr,rf,svr,lstm,gru]")
	parser.add_argument("--metric", type=str, default="mse", help = "Available options : [mse,mae,r2,all]")  
	parser.add_argument("--predict", type=boolean_string, default=False,help = "True -> Predicts on the testing data and prints the evaluation metrics, False -> Does not predict on testing data")
	parser.add_argument("--plot_results", type=boolean_string, default=False, help = "Plots the predictions (if True switches --predict to True automatically)")
	parser.add_argument("--save", type=boolean_string, default=False, help = "Saves the predictions (if True switches --predict to True automatically)")   
	parser.add_argument("--skip_clean", type=boolean_string, default=False, help = "Assumes you have the final dataset and skips the data cleaning.. If you are in windows, then download the cleaned dataset online and then set this parameter to True")        
	parser.add_argument("--clean_split", type=boolean_string, default=True, help = "True -> Performs cleaning and splitting, False -> Skips cleaning and splitting (only use it for debugging purposes)")   
	parser.add_argument("--per_change", type=boolean_string, default=True, help = " Altered arc formula of change in price, the norm can be set with the parameter norm") 
	parser.add_argument("--norm", type=int, default=1000, help = "Norm of --per_change, Look at the paper for more details. Recommendation : If the changes between subsequent prices are high (i.e the interval of recording is large), use a smaller norm ")			     
	
	args = parser.parse_args()
	
	return args


def main():


	args = get_args()
	spline = args.spline
	plot_imputation = args.plot_imputation
	train_size = args.train_size
	price_only = args.price_only
	shift_param = args.shift_param
	model_type = args.model_type.lower()
	metric = args.metric.lower()
	predict = args.predict
	save = args.save
	plot_results = args.plot_results
	clean_split = args.clean_split
	skip_clean = args.skip_clean
	norm = args.norm
	per_change = args.per_change
	
	if price_only == False and model_type == "svr":
		raise Exception("SVR only works when price only is set to True")
	if price_only == False and per_change == True:
		raise Exception("the per_change option only works when using the price only as a feature")

	if skip_clean:
		dataSplitting.main(train_size,price_only,shift_param,per_change)
	elif clean_split:
		dataPreprocessing.main(spline,plot_imputation,norm)
		dataSplitting.main(train_size,price_only,shift_param,per_change)


	if model_type in ["rf","lr","dtr","xgboost","svr"]:
		import trees_LR_SVR
		model = trees_LR_SVR.main(model_type,metric,predict,save,plot_results,train_size, norm, per_change)
		
	elif model_type in ["lstm","gru"]:
		import LSTM_GRU
		model = LSTM_GRU.main(model_type,metric,predict,save,plot_results,train_size, norm, per_change)
	else:
		raise ValueError("unknown model " + str(self.name) + ", choose one of the following [lstm,gru,rf,lr,xgboost,dtr,svr] ")



if __name__ == '__main__':
    main()
