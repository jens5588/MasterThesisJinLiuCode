import pandas as pd
import numpy as np
import keras
import os
import time
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import ParameterGrid
from numpy.random import seed
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import json
from collections import Counter
import warnings
from os import path
from joblib import Parallel, delayed
import multiprocessing

seed(123)
tensorflow.random.set_seed(123)


class MetricsValidation(keras.callbacks.Callback):
    def __init__(self, model, validation_data, selectedfeatures, lastprices, verbose):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        self.selectedfeatures = selectedfeatures
        self.lastprices = lastprices
        self.verbose = verbose

    # def on_train_begin(self, logs={}):
    #    self.iteratedmae = 0

    def on_epoch_end(self, epoch, logs={}):
        validationdays = int(self.validation_data[0].shape[0] / 24)
        y_hatlist = []
        for validationday in range(validationdays, 0, -1):
            yhat_dailylist = []
            if validationday > 1:
                validationdaily = self.validation_data[0][-24 * validationday:-24 * (validationday - 1)]
            else:
                validationdaily = self.validation_data[0][-24 * validationday:]
            lastprices = self.lastprices[-24 * (validationday + 1):-24 * validationday].flatten().tolist()
            lastprices.reverse()

            for hour in range(24):
                pricedict = {}
                for lag in range(1, 25):
                    pricedict['price difference t_' + str(lag)] = lastprices[lag - 1]
                validation_x = validationdaily[hour].flatten().reshape(1, -1)
                if len(yhat_dailylist) > 0:
                    for feature in self.selectedfeatures:
                        if feature in pricedict.keys():
                            featureindex = self.selectedfeatures.index(feature)
                            validation_x[0, featureindex] = pricedict[feature]
                yhat = float(self.model.predict(validation_x))
                yhat_dailylist.append(yhat)
                lastprices.insert(0, yhat)
                del (lastprices[-1])
            y_hatlist.append(np.asarray(yhat_dailylist))
        y_pred = np.asarray(y_hatlist).flatten()
        y_true = np.asarray(self.validation_data[1][-24 * validationdays:]).flatten()
        m = tensorflow.keras.metrics.MeanAbsoluteError()
        iteratedmae = m(y_pred, y_true)
        logs['iteratedmae'] = iteratedmae
        if self.verbose:
            print('-val-iterated: %f' % np.round(iteratedmae, decimals=4))


def miselection(df, feature_number, allfeatures):
    y = df['price difference']
    X = df[allfeatures]
    mi = mutual_info_regression(X[:-24], y[:-24], n_neighbors=5)
    mi /= np.max(mi)
    dictionary = dict(zip(allfeatures, mi))
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [sorted_dict[i][0] for i in range(len(sorted_dict))]
    selected_features = sorted_features[:feature_number]
    return selected_features


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


def modelbuilding(hiddenlayersizes, featurenum, activation, dropout):
    model = Sequential()
    # if inputlayer dropout
    # model.add(Dropout(0.1,inputshape=()))
    # model.add(Dense(hiddenlayersizes[0],init=...))
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
    model.add(Dense(1, activation='linear'))
    return model


def trainandpredict(totaldata, scaler, selectedfeatures, validationdays, testdays, hiddenlayersizes, dropout,
                    activation, epochs, batch_size, patience, verbose):
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
    test = totaldata[-int(testdays * 24):, :]
    featurenum = len(selectedfeatures)
    model = modelbuilding(hiddenlayersizes=hiddenlayersizes, featurenum=featurenum, activation=activation,
                          dropout=dropout)
    # print(model.summary())
    model.compile(loss='mae', optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), metrics=['mae'])
    lastprices = scaled_y[-24 * (validationdays + 1):]
    # metricsvalidation = MetricsValidation(model=model, validation_data=(validation_x, validation_y),
    #                                    selectedfeatures=selectedfeatures, lastprices=lastprices, verbose=verbose)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(validation_x, validation_y), callbacks=[es])
    y_hatlist = []
    for testday in range(testdays, 0, -1):
        yhat_dailylist = []
        if testday > 1:
            testdaily = test[-int(24 * testday):-int(24 * (testday - 1)), :]
        else:
            testdaily = test[-int(24 * testday):, :]
        lastprices = totaldata[-int(23 + testday * 24):-int(testday * 24), 0].tolist()
        lastprices.reverse()

        for hour in range(24):
            pricedict = {}
            # the last lagged residul (price difference) are changed for each hour. The predicted values is used instead
            # of real values. If a new value is predicted, it is added to the lastprices list
            for lag in range(1, 24):
                pricedict['price difference t_' + str(lag)] = lastprices[lag - 1]
            test_x = testdaily[hour, 1:].reshape(1, -1)
            if len(yhat_dailylist) > 0:
                for feature in selectedfeatures:
                    if feature in pricedict.keys():
                        featureindex = selectedfeatures.index(feature)
                        test_x[0, featureindex] = pricedict[feature]
            scaled_x = scaler_x.transform(test_x)
            yhat = model.predict(scaled_x)
            yhat = float(scaler_y.inverse_transform(yhat))
            yhat_dailylist.append(yhat)
            lastprices.insert(0, yhat)
            del (lastprices[-1])
        y_hatlist.append(np.asarray(yhat_dailylist))
    y = test[:, 0].tolist()

    return np.asarray(y_hatlist).flatten(), np.asarray(y)


def run(daylist):
    yhatlist = []
    ylist = []
    timestamplist = []
    sectionbegin = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=daylist[0])
    sectionbegin = sectionbegin.strftime('%Y-%m-%d')
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
                                          selectedfeatures=selectedfeatures,
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'None':
                selectedfeatures = features
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          selectedfeatures=selectedfeatures,
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'mrmrensemble':
                # selectedfeaturelist = mrmrensembleselection(datacombined, featurenumber, features)
                selectedfeaturelist = selectedfeaturedict[datum]
                y_hatensemblelist = []
                for selectedfeatures in selectedfeaturelist[:1]:
                    tmp = datacombined[['price difference'] + selectedfeatures].values
                    yhat, y = trainandpredict(totaldata=tmp, scaler=scaler, selectedfeatures=selectedfeatures,
                                              validationdays=validationdays, testdays=testdays,
                                              hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                              activation=activation, epochs=epochs, batch_size=batchsize,
                                              patience=patience, verbose=verbose)
                    y_hatensemblelist.append(yhat)
                yhat = np.mean(y_hatensemblelist, axis=0)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
    else:
        for i in [daydeltalist[j * 7] for j in range(int(len(daydeltalist) / 7))]:
            predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i + 6)
            timestamps, datacombined = collectingdata(inputdftmp, predictiondate, lookbackyears,
                                                      trainingdayscurrent, trainingdaysprevious, testdays)
            if featureselection == 'mi':
                selectedfeatures = miselection(datacombined, featurenumber, features)
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          selectedfeatures=selectedfeatures,
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'None':
                selectedfeatures = features
                datacombined = datacombined[['price difference'] + selectedfeatures].values
                yhat, y = trainandpredict(totaldata=datacombined, scaler=scaler,
                                          selectedfeatures=selectedfeatures,
                                          validationdays=validationdays, testdays=testdays,
                                          hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                          activation=activation, epochs=epochs, batch_size=batchsize,
                                          patience=patience, verbose=verbose)
                timestamplist.append(timestamps)
                yhatlist.append(yhat)
                ylist.append(y)
            if featureselection == 'mrmrensemble':
                # selectedfeaturelist = mrmrensembleselection(datacombined, featurenumber, features)
                selectedfeaturelist = selectedfeaturedict['datum']
                y_hatensemblelist = []
                for selectedfeatures in selectedfeaturelist:
                    tmp = datacombined[['price difference'] + selectedfeatures].values
                    yhat, y = trainandpredict(totaldata=tmp, scaler=scaler, selectedfeatures=selectedfeatures,
                                              validationdays=validationdays, testdays=testdays,
                                              hiddenlayersizes=hiddenlayersizes, dropout=dropout,
                                              activation=activation, epochs=epochs, batch_size=batchsize,
                                              patience=patience, verbose=verbose)
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
    filepath = path.join('Prediction/RecursiveMLP', filename + '.' + 'csv')
    predictiondf.to_csv(filepath, sep=';', index=False)


if __name__ == '__main__':
    with open('Config/config_recursivemlp.json', 'r') as fp:
        parameterdict = json.load(fp)
    inputfilepath = parameterdict['inputfilepath']
    inputdf = pd.read_csv(inputfilepath, sep=';', index_col=['UTC Timestamp'])
    predictionbegin = parameterdict['predictionbegin']
    daystopredict = parameterdict['daystopredict']
    validationdays = parameterdict['validationdays']
    testdays = parameterdict['testdays']
    runtimes = parameterdict['runtimes']
    verbose = parameterdict['verbose']

    # parameters for the deep neural networks
    hyperparameters = parameterdict['hyperparameters']
    params_grid = ParameterGrid(hyperparameters)

    # read json file for days to predict
    with open('Data/sampledays2018.json', 'r') as fp:
        daylist = json.load(fp)
    daydeltalist = [
        int((datetime.strptime(item, "%Y-%m-%d") - datetime.strptime(predictionbegin,
                                                                     "%Y-%m-%d")).total_seconds() / 86400)
        for item in daylist]

    # with open('bench_all24_selectedfeature_mrmrensemble.json', 'r') as fp:
    #    selectedfeaturedict = json.load(fp)
    selectedfeaturedict = {}
    for enum, params in enumerate(tqdm(params_grid)):
        lags = params['lags']
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
        # add the features with the p consecutive lagged variable
        features = parameterdict['features'] + ['price difference t_' + str(i) for i in range(1, lags + 1)]
        featurenumber = parameterdict['featurenumber']
        allcolumns = ['price difference'] + features
        inputdftmp = inputdf[allcolumns]
        remarks = 'Iterative' + 'LB' + str(lookbackyears) + 'VD' + str(validationdays) + 'Lag' + str(
            lags) + 'hiddenL' + str(len(hiddenlayersizes)) + 'S' + str(
            hiddenlayersizes[0]) + featureselection

        #Parallel(n_jobs=-1)(delayed(run)(item) for item in
        #                    [list(range(0, 90)), list(range(90, 180)), list(range(180, 270)), list(range(270, 365))])
        run(list(range(365)))
