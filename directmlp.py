import pandas as pd
import numpy as np
import keras
import os
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import model_from_json
from numpy.random import seed
from tqdm import tqdm
from datetime import datetime, timedelta
import json
from os import path
import time
from joblib import Parallel, delayed
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri as rpyn
from collections import Counter

seed(123)
tensorflow.random.set_seed(123)

def mrmrensembleselection(dataframe, featurenumber, allfeatures):
    columnslist = ['price difference'] + allfeatures
    dataframe = dataframe[columnslist].dropna().reset_index(drop=True)
    dataframe = dataframe[:-24]
    dfcolumns = dataframe.columns.tolist()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataframe.values)
    dataframe = pd.DataFrame(scaled, columns=dfcolumns)
    with localconverter(ro.default_converter + pandas2ri.converter):
        tmp = ro.conversion.py2rpy(dataframe)
    # feature count in the rstring must be manually adjusted, equals the number of features to be selected
    rstring = """
    function(dataframe){
        library(mRMRe)
        dataframe <- sapply(dataframe, as.numeric)
        feature_data <- mRMR.data(data=data.frame(dataframe))
        featureNames(feature_data)
        en <- mRMR.ensemble(data=feature_data,target_indices=c(1),solution_count=1,feature_count=37)
        solutions(en)
        }
        """
    rfunc = ro.r(rstring)
    en = rfunc(tmp)
    selectedindex = [en[0][i] for i in range(int(1 * featurenumber))]
    selectedfeatures = [columnslist[item - 1] for item in selectedindex]
    featurelist = []
    for i in range(1):  # equals solution_count, feature_count equals featurenumber
        featurelist.append(
            [selectedfeatures[j] for j in range(int(i * featurenumber), int(i * featurenumber + featurenumber))])

    return featurelist

# feature selection based on mutual information
def miselection(df, feature_number, allfeatures):
    y = df['price difference']
    X = df[allfeatures]
    mi = mutual_info_regression(X[:-365], y[:-365], n_neighbors=10)
    mi /= np.max(mi)
    dictionary = dict(zip(allfeatures, mi))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [sorted_dict[i][0] for i in range(len(sorted_dict))]
    selected_features = sorted_features[:feature_number]
    return selected_features

# generate training, validation and test data
def collectingdata(inputdf, predictiondate, lookbackyears, trainingdayscurrent, trainingdaysprevious, testdays):
    datebegincurrent = predictiondate + timedelta(days=-(trainingdayscurrent))
    datacurrent = inputdf.loc[(datebegincurrent.strftime('%Y-%m-%d') + ' 00'):(
            predictiondate.strftime('%Y-%m-%d') + ' 23'), :]
    timestamps = datacurrent.index.to_list()[-int(24 * testdays):]
    if lookbackyears > 0:
        combinedprevious = None
        for year in range(lookbackyears, 0, -1):
            datebeginprevious = predictiondate + timedelta(days=-(trainingdaysprevious + 365 * year))
            dateendprevious = datebeginprevious + timedelta(days=2 * trainingdaysprevious - 1)
            dataprevious = inputdf.loc[(datebeginprevious.strftime('%Y-%m-%d') + ' 00'):(
                    dateendprevious.strftime('%Y-%m-%d') + ' 23'), :]
            if combinedprevious is None:
                combinedprevious = dataprevious
            else:
                combinedprevious = pd.concat([combinedprevious, dataprevious], ignore_index=True)
        datacombined = pd.concat([combinedprevious, datacurrent], ignore_index=True).dropna()
    else:
        datacombined = datacurrent.dropna()
    return timestamps, datacombined

# build the direct mlp model
def modelbuilding(hiddenlayersizes, featurenum, activation, dropout):
    model = Sequential()
    model.add(
        Dense(hiddenlayersizes[0], input_dim=featurenum, kernel_initializer='normal', activation=activation))
    if dropout > 0:
        model.add(Dropout(dropout, seed=123))
    for layersize in hiddenlayersizes[1:]:
        if dropout > 0:
            model.add(Dense(layersize, activation=activation))
            model.add(Dropout(dropout, seed=123))
        else:
            model.add(Dense(layersize, activation=activation))
    # for regression problem, last layer uses linear activation function
    model.add(Dense(1, activation='linear'))
    return model

# model training, validation and prediction
def trainandpredict(totaldata, scaler, featuredimension, validationdays, testdays, hiddenlayersizes, dropout,
                    activation, epochs, batch_size, patience, verbose, runtimes):
    if scaler == 'MinMaxScaler':
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    scaled_x = scaler_x.fit_transform(totaldata[:-int(testdays * 24), 1:])
    scaled_y = scaler_y.fit_transform(totaldata[:-int(testdays * 24), 0].reshape(-1, 1))
    train_x = scaled_x[:-int(validationdays * 24), :]
    train_y = scaled_y[:-int(validationdays * 24), :]
    validation_x = scaled_x[-int(validationdays * 24):, :]
    validation_y = scaled_y[-int(validationdays * 24):, :]
    test_x = totaldata[-int(testdays * 24):, 1:]
    test_y = totaldata[-int(testdays * 24):, 0]
    scaled_testx = scaler_x.transform(test_x)
    y_hatlist = []

    for i in range(runtimes):
        model = modelbuilding(hiddenlayersizes=hiddenlayersizes, featurenum=featuredimension, activation=activation,
                              dropout=dropout)
        # print(model.summary())
        model.compile(loss='mae', optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), metrics=['mae'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience,
                           restore_best_weights=True)
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                  validation_data=(validation_x, validation_y), callbacks=[es])

        yhat = model.predict(scaled_testx)
        # transform the predicted value back
        yhat = scaler_y.inverse_transform(yhat.reshape(-1, 1)).flatten()
        y_hatlist.append(yhat)

    yhat = np.asarray(y_hatlist).mean(axis=0)
    y = test_y
    return yhat, y


def run(daylist):
    yhatlist = []
    ylist = []
    timestamplist = []
    sectionbegin = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=daylist[0])
    sectionbegin = sectionbegin.strftime('%Y-%m-%d')
    # number of test days determine model update frequency, we update the model daily, so traindays is 1
    if testdays == 1:
        for i in tqdm(daylist):
            predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i)
            datum = predictiondate.strftime('%Y-%m-%d')
            timestamps, datacombined = collectingdata(inputdftmp, predictiondate, params['lookbackyears'],
                                                      params['trainingdayscurrent'], params['trainingdaysprevious'],
                                                      testdays)
            if featureselection == 'mi':
                selectedfeatures = miselection(datacombined, featurenumber, features)
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          featuredimension=len(selectedfeatures),
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose, runtimes=runtimes)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'None':
                selectedfeatures = features
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          featuredimension=len(selectedfeatures),
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose, runtimes=runtimes)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'mrmrensemble':
                selectedfeaturelist = mrmrensembleselection(datacombined, featurenumber, features)
                #selectedfeaturelist = selectedfeaturedict[datum]
                y_hatensemblelist = []
                for selectedfeatures in selectedfeaturelist[:1]:
                    tmp = datacombined[['price difference'] + selectedfeatures].values
                    yhat, y = trainandpredict(totaldata=tmp, scaler=scaler, featuredimension=len(selectedfeatures),
                                              validationdays=validationdays, testdays=testdays,
                                              hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                              activation=activation, epochs=epochs, batch_size=batchsize,
                                              patience=patience, verbose=verbose, runtimes=runtimes)
                    y_hatensemblelist.append(yhat)
                yhat = np.mean(y_hatensemblelist, axis=0)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
    else:
        for i in [daylist[j * 7] for j in range(int(len(daylist) / 7))]:
            predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i + 6)
            datum = predictiondate.strftime('%Y-%m-%d')
            timestamps, datacombined = collectingdata(inputdftmp, predictiondate, lookbackyears,
                                                      trainingdayscurrent, trainingdaysprevious, testdays)
            if featureselection == 'mi':
                selectedfeatures = miselection(datacombined, featurenumber, features)
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          featuredimension=len(selectedfeatures),
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose, runtimes=runtimes)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'None':
                selectedfeatures = features
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          featuredimension=len(selectedfeatures),
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose, runtimes=runtimes)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'mrmrensemble':
                selectedfeaturelist = mrmrensembleselection(datacombined, featurenumber, features)
                #selectedfeaturelist = selectedfeaturedict[datum]
                y_hatensemblelist = []
                for selectedfeatures in selectedfeaturelist:
                    tmp = datacombined[['price difference'] + selectedfeatures].values
                    yhat, y = trainandpredict(totaldata=tmp, scaler=scaler, featuredimension=len(selectedfeatures),
                                              validationdays=validationdays, testdays=testdays,
                                              hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                              activation=activation, epochs=epochs, batch_size=batchsize,
                                              patience=patience, verbose=verbose, runtimes=runtimes)
                    y_hatensemblelist.append(yhat)
                yhat = np.mean(y_hatensemblelist, axis=0)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)

    predictionlist = np.asarray(yhatlist).flatten()
    actuallist = np.asarray(ylist).flatten()
    timestamplist = np.asarray(timestamplist).flatten()
    error = actuallist - predictionlist
    predictiondf = pd.DataFrame(
        {'timestamp': timestamplist, 'prediction': predictionlist, 'actual': actuallist, 'error': error})
    filename = sectionbegin + '_' + remarks
    filepath = path.join('Prediction/DirectMLP', filename + '.' + 'csv')
    predictiondf.to_csv(filepath, sep=';', index=False)


if __name__ == '__main__':
    with open('Config/config_directmlp.json', 'r') as fp:
        parameterdict = json.load(fp)
    inputfilepath = parameterdict['inputfilepath']
    inputdf = pd.read_csv(inputfilepath, sep=';', index_col=['UTC Timestamp'])
    predictionbegin = parameterdict['predictionbegin']
    daystopredict = parameterdict['daystopredict']
    validationdays = parameterdict['validationdays']
    testdays = parameterdict['testdays']
    runtimes = parameterdict['runtimes']
    verbose = parameterdict['verbose']

    hyperparameters = parameterdict['hyperparameters']
    params_grid = ParameterGrid(hyperparameters)

    # read json file for days to predict
    with open('Data/sampledays2018.json', 'r') as fp:
        daylist = json.load(fp)
    # one day 86400 seconds
    daydeltalist = [
        int((datetime.strptime(item, "%Y-%m-%d") - datetime.strptime(predictionbegin,
                                                                     "%Y-%m-%d")).total_seconds() / 86400)
        for item in daylist]




    for enum, params in enumerate(tqdm(params_grid)):
        starttime = time.time()
        # hyperparameters
        lookbackyears = params['lookbackyears']
        trainingdayscurrent = params['trainingdayscurrent']
        trainingdaysprevious = params['trainingdaysprevious']
        featureselection = params['featureselection']
        scaler = params['scaler']
        hiddenlayersizes = params['hiddenlayersizes']
        dropout = params['dropout']
        activation = params['activation']
        patience = params['patience']
        epochs = params['epochs']
        batchsize = params['batchsize']
        features = parameterdict['features']
        featurenumber = parameterdict['featurenumber']
        allcolumns = ['price difference'] + features
        inputdftmp = inputdf[allcolumns]
        remarks = 'DirectMLP' + 'LB' + str(lookbackyears) + 'VD' + str(validationdays) + 'hiddenL' + \
                  str(len(hiddenlayersizes)) + 'S' + str(hiddenlayersizes[0])
        # Parallel is for multiprocessing. Since forecasting each day is independent, multiprocessing is possible
        # Parallel(n_jobs=-1)(delayed(run)(item) for item in
        # [list(range(90)), list(range(90, 180)), list(range(180, 270)), list(range(270, 365))])
        run(list(range(daystopredict)))
