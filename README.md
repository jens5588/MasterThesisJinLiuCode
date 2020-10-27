# Important tips
## 1. Project componets
### 1.1 Data
In the data directory, the orignial input file is 'Table_Fundamentalmodell_InOut.xlsx' and 'holidays'.
The 'preprocessed price.csv' is generated with Python script 'preprocessing.py', which shifts the time series variables 
to generate lagged variables, and adds time relevant variables like hour, weekday, holiday. Last but not least, 
preprocessing of auxiliary variables. The '2013-2018Label.csv' file is genarated with 'predictionresultslabelling.py'.
In this file contains the information of all predictions of different models. And the prediction errors, final submitted 
electricity prices (fundamental model + residual forecsting) are included. The labelling function labels the best model 
in each hour, 0 for direct MLP, 1 for SARIMAX, 2 for recursive MLP and 3 for Encoder-Decoder. The 'sampledays2018.json' 
contains the days selected in 2018 for hyperparameter optimization of the neural network based models. The 
sampledays2018 is generated with function 'generatedays' in 'analytics.py'
### 1.2 Config
The config directory contains three config file. Each config file contains the corresponding parameters and 
hyperparameters for the neural network models. The parameter 'verbose' indicates whether the training process shoudld be 
printed out. 0 for no print, 1 for print.
### 1.3 FiguresInTheThesis
This directory contains the figures in the thesis. These figures are generated with the file 'FiguresInTheThesis.ipynb',
which could be excuted with Jupyter Notebook
### 1.4 Prediction
The prediction directory contains the residual prediction of different models. Each model directory contains predictions
from 2013 to 2018. And in each .csv file contains the actual residual value (actual), predicted residual value
(prediction), and the error of the prediction (error).
### 1.5 Visualisation
The visualisation directory contains basic visualisation of the data. The code for visualisation is in the 
'visualisation.py' file
## 2. Code for each model
### 2.1 Naive model
We have for each model a code file. The naive model does not need specific code. We have the results direct generated 
'predictionresultslabelling' file.
### 2.2 direct mlp & recursive mlp model
For both models, feature selection options are considered. There are two specific feature selection method, miselection 
and mrmrensemble selection. The mrmrensemble method is a package from R. We call the R function in Python with the
rstring. So, if the mrmrensemble is used to select variables, the line  
'en <- mRMR.ensemble(data=feature_data,target_indices=c(1),solution_count=1,feature_count=37)' must be manually changed. 
feature_count is the number of features to be selected. The default value for feature selection is 'None', which means 
no further feature selection from the 'features' in the config file. The features in the corresponding config file are 
also the final features used in the final predictions.
The both model could be multiprocessed, which mean forecasting several days together. With the 
'Parallel(n_jobs=-1)(delayed(run)(item) for item in [list(range(90)), list(range(90, 180)), list(range(180, 270)), 
list(range(270, 365))])' the forecasting for one year could be excuted in four processes.
If only with 'run(list(range(365)))', multiprocessing is not used.
### 2.3 Encoder-Decoder model
The encoder-decoder model needs more time for excution, if there is no GPU accleration. To reduce the running time, 
one option is to reduce the patience in the config file. So the total excuted epochs would be reduced.
### 2.4 sarimax model
The sarimax model contains the grid search process. In each day, nine models are fitted. The search process needs about 
10 minutes. So for one year prediction, it need 60-70 hours.
### 2.5 knn model
The knn model need very little time. For each year, only 10 minutes are needed.




