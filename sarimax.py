import numpy as np
import pandas as pd
import json
import time
from itertools import product
import statistics
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import scipy.stats as scs
from itertools import product
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from os import path
import warnings

warnings.filterwarnings('ignore')




def sarimaxAIC(totaldata, paramlist):
    best_aic = float("inf")
    train_endog = totaldata[:-int(24 * testdays), 0]
    train_exog = totaldata[:-int(24 * testdays), 1:]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_endog = scaler_y.fit_transform(train_endog.reshape(-1, 1))
    train_exog = scaler_x.fit_transform(train_exog)
    test_exog = scaler_x.transform(totaldata[-int(24 * testdays):, 1:])
    y = totaldata[-24:, 0]
    for params in tqdm(paramlist):
        try:
            model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog,
                                              order=(params[0], params[1], params[2]),
                                              seasonal_order=(params[3], params[4], params[5], params[6])).fit(disp=0)
        except:
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = params

    pvalues = best_model.pvalues[:len(columnused) - 1].tolist()
    yhat = best_model.predict(start=train_endog.shape[0], end=train_endog.shape[0] + int(24 * testdays - 1),
                              exog=test_exog)
    yhat = scaler_y.inverse_transform(yhat.reshape(-1, 1))

    return y, yhat, best_param, pvalues


def simpleAR(totaldata, parameterslist):
    traindata = totaldata[:-int(24 * testdays), 0]
    model = sm.tsa.statespace.SARIMAX(traindata, order=(parameterslist[0], parameterslist[1], parameterslist[2])
                                      ).fit(disp=-1)
    yhat = model.predict(start=traindata.shape[0], end=traindata.shape[0] + int(24 * testdays - 1))
    return yhat


def ARX(totaldata, parameterslist):
    train_endog = totaldata[:-int(24 * testdays), 0]
    train_exog = totaldata[:-int(24 * testdays), 1:]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_endog = scaler_y.fit_transform(train_endog.reshape(-1, 1))
    train_exog = scaler_x.fit_transform(train_exog)
    test_exog = scaler_x.transform(totaldata[-int(24 * testdays):, 1:])
    model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog,
                                      order=(parameterslist[0], parameterslist[1], parameterslist[2])).fit(disp=0)
    yhat = model.predict(start=train_endog.shape[0], end=train_endog.shape[0] + int(24 * testdays - 1),
                         exog=test_exog)

    return yhat


def sarimax(totaldata, params):
    train_endog = totaldata[:-int(24 * testdays), 0]
    train_exog = totaldata[:-int(24 * testdays), 1:]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_endog = scaler_y.fit_transform(train_endog.reshape(-1, 1))
    train_exog = scaler_x.fit_transform(train_exog)
    test_exog = scaler_x.transform(totaldata[-int(24 * testdays):, 1:])
    model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog,
                                      order=(params[0], params[1], params[2]),
                                      seasonal_order=(params[3], params[4], params[5], params[6])).fit(disp=0)
    yhat = model.predict(start=train_endog.shape[0], end=train_endog.shape[0] + int(24 * testdays - 1), exog=test_exog)
    yhat = scaler_y.inverse_transform(yhat.reshape(-1, 1))
    return yhat


def run(daylist):
    results = []
    ylist = []
    yhatlist = []
    timestamplist = []
    pvaluesdict = {}
    sectionbegin = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=daylist[0])
    sectionbegin = sectionbegin.strftime('%Y-%m-%d')
    for i in tqdm(daylist):
        predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i)
        datum = predictiondate.strftime('%Y-%m-%d')
        traindatebegin = predictiondate + timedelta(days=-(trainingdays))
        totaldata = inputdf.loc[
                    (traindatebegin.strftime('%Y-%m-%d') + ' 00'):(predictiondate.strftime('%Y-%m-%d') + ' 23'), :]
        timestamps = totaldata.index.to_list()[-int(24 * testdays):]
        timestamplist = timestamplist + timestamps
        datavalues = totaldata.values
        # Augmented Dickey-Fuller unit root test
        p_value = adfuller(datavalues[:-24, 0])[1]
        # if p greater than critical value, differencing
        if p_value > 0.05:
            d_tmp = range(1, 2)
            parameterslist = list(product(p, d_tmp, q, Ps, Ds, Qs, S))
            y, yhat, best_param, pvalues = sarimaxAIC(datavalues, parameterslist)
        else:
            parameterslist = list(product(p, d, q, Ps, Ds, Qs, S))
            y, yhat, best_param, pvalues = sarimaxAIC(datavalues, parameterslist)
        ylist.append(y)
        yhatlist.append(yhat)
        results.append([datum, best_param])
        pvaluesdict[datum] = pvalues

    yhatlist = np.concatenate(yhatlist).ravel().tolist()
    ylist = np.concatenate(ylist).ravel().tolist()
    error = [ylist[i] - yhatlist[i] for i in range(len(yhatlist))]
    filename = sectionbegin + remarks
    filepath = path.join('Prediction/SARIMAX', filename + '.' + 'csv')
    predictiondf = pd.DataFrame(
        {'timestamp': timestamplist, 'prediction': yhatlist, 'actual': ylist, 'error': error})
    predictiondf.to_csv(filepath, sep=';', index=False)
    resultsdf = pd.DataFrame(data=results, columns=['Datum', 'Parameter'])
    tablename = sectionbegin + remarks + 'AICTable'
    tablepath = path.join('Prediction/SARIMAX', tablename + '.' + 'csv')
    resultsdf.to_csv(tablepath, sep=';', index=False)
    jsonfilepath = path.join('Prediction/SARIMAX', sectionbegin + remarks + 'pvalues.json')
    with open(jsonfilepath, 'w') as fp:
        json.dump(pvaluesdict, fp)


if __name__ == '__main__':
    inputdf = pd.read_csv('Data/preprocessed price.csv', sep=';', index_col=['UTC Timestamp'])
    columnused = ['price difference', 'Holiday', 'daily temperature', 'PV Prediction', 'price estimator Model',
                  'Demand', 'Wind Prediction', 'Weekday', 'Hour', 'NTC NET', "price difference t_48",
                  "price difference t_72", "price difference t_96", "price difference t_120", "price difference t_144",
                  "price difference t_168", "price difference t_336", "price difference t_504"]
    inputdf = inputdf[columnused]
    predictionbegin = '2018-01-01'
    daystopredict = 365
    trainingdays = 49
    testdays = 1
    p = range(6, 12, 2)
    d = range(0, 1)
    q = range(0, 1)
    Ps = range(1, 2)
    Ds = range(0, 1)
    Qs = range(0, 3)
    S = range(24, 25)
    remarks = 'SARIMAXAICpQ'
    run(list(range(daystopredict)))
