# Get Started
1. Install the Dataset from this link https://www.kaggle.com/mczielinski/bitcoin-historical-data
2. Download all the resources from the repository
3. Inside the folder Datasets add the downloaded ./Datasets and name it : "bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv"
4. If you are using a Linux or Mac environment you can skip step 5 and 6
5. If you are using a windows environment, the dataPreprocessing.py program will not work properly, because the pandas library is still not fully supported on windows environments, so you will need to manually download the cleaned dataset, name it "final_Bitcoin_dataset.csv" and save it inside the ./Datasets folder from the link : https://mega.nz/folder/bph2GCDB#0jfmGQBvYkOR-e8BgGo-ig
6. If you are using a windows environment, you will need to always pass the parameter --skip_clean True
7. You are all set, Enjoy !

# Basics on how to run :
To have a quick test, execute :
```bash
python main.py --plot_results True --model_type lr
```

# Available parameters
To have a detailed look at the available parameters, execute :
```bash
python main.py -h
```

# Changes to be expected  :
1. Creation of csv file inside the ./Datasets folder of the final cleaned dataset
2. Creation of split folder containing the training and testing data
3. Creation of .npy files in ./Results if --save == True


# Load Existing weights for the RNN models
To load existing model weights, go to LSTM_GRU.py
Comment :
```python
history = lstm.fit(self.X_train, self.y_train, epochs=10, batch_size=3, validation_data=(self.X_test, self.y_test), verbose=True, shuffle=False)
lstm.save("./lstm_weights.h5")
```
Uncomment :
```python
lstm.load_weights("./lstm_weights.h5")
```
Availabe weights : "lstm_weights_updatedMethod.h5" (using proposed method), lstm_no_normalization.h5 (not using normalization)

# Collaborators
	- Amraoui Elarbi
	- Chenjie Yang
	- Zepu Wang

# License
Copyright: (c) 2021 Elarbi Amraoui
		   (c) 2021 Chenjie Yang
		   (c) 2021 Zepu Wang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

