# The code for future inputs in the decoder part is inspired by the blog of Alessandro Angioi. Cited in the thesis
# url: https://www.angioi.com/time-series-encoder-decoder-tensorflow/

import numpy as np
import json
import keras
import time
import tensorflow as tf
import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from keras.optimizers import Adam
from datetime import datetime
from datetime import timedelta
from numpy.random import seed
from joblib import Parallel, delayed
import multiprocessing

seed(123)
tf.random.set_seed(123)




def createDataset(data, label, features, n_insteps=24, n_outsteps=24, daydiscretization=False):
    allcolumns = label + features
    df = pd.DataFrame(data, columns=allcolumns)
    for i in range(n_insteps - 1, 0, -1):
        for column in allcolumns:
            df[column + '(l_%d)' % i] = df[column].shift(i)
    columns = df.columns.to_list()
    columns = columns[len(allcolumns):] + columns[:len(allcolumns)]
    df = df[columns]
    # create future features inputs
    for i in range(1, n_outsteps + 1):
        for column in features:
            df[column + '(f_%d)' % i] = df[column].shift(-i)
    # create future input variables to predict price difference, f stands for future
    for i in range(1, n_outsteps + 1):
        for column in label:
            df[column + '(f_%d)' % i] = df[column].shift(-i)

    df = df.dropna().reset_index(drop=True)
    # if daydiscretization is true, then training data for encoder only begins from hour 0 are used, this reduces
    # total training data number. By default daydiscretization is 0, which could generate more training data. This means
    # that training data for encoder could also begin from hour 1 or hour 2 etc.
    if daydiscretization:
        daysnumber = int(df.shape[0] / n_insteps)
        l = [int(i * 24) for i in range(daysnumber + 1)]
        df = df.loc[l, :].reset_index(drop=True)
    return df


def collectingdata(inputdf, n_insteps, n_outsteps, predictiondate, lookbackyears, trainingdayscurrent,
                   trainingdaysprevious, testdays):
    datebegincurrent = predictiondate + timedelta(days=-(trainingdayscurrent + n_insteps / 24))
    datacurrent = inputdf.loc[(datebegincurrent.strftime('%Y-%m-%d') + ' 00'):(
            (predictiondate).strftime('%Y-%m-%d') + ' 23'), :]
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
    return timestamps, datacombined.values.astype('float32')


def trainandpredict(totaldata, scaler, n_insteps, n_outsteps, featuredimension, label, daydiscretization,
                    validationdays, testdays, hiddenunits, activation, epochs, batch_size, patience, runtimes, verbose):
    if scaler == 'MinMaxScaler':
        scaler_pastinputs = MinMaxScaler(feature_range=(0, 1))
        scaler_futureinputs = MinMaxScaler(feature_range=(0, 1))
        scaler_label = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler_pastinputs = StandardScaler()
        scaler_futureinputs = StandardScaler()
        scaler_label = StandardScaler()


    if not daydiscretization:
        scaled_pastinputs = scaler_pastinputs.fit_transform(
            totaldata[:-int(testdays * 24), :(featuredimension + len(label)) * n_insteps])
        scaled_futureinputs = scaler_futureinputs.fit_transform(
            totaldata[:-int(testdays * 24),
            (featuredimension + len(label)) * n_insteps: (featuredimension + len(label)) *
                                                         n_insteps + featuredimension * n_outsteps])

        scaled_label = scaler_label.fit_transform(totaldata[:-int(testdays * 24), -n_outsteps:])
        train_pastinputs = scaled_pastinputs[:-int(validationdays * 24), :]
        # 3D [samples, timesteps, features]
        train_pastinputs = train_pastinputs.reshape(train_pastinputs.shape[0], n_insteps, featuredimension + len(label))
        # future inputs are something determined in the future (next 24 hour), like X in the ARX model, could be
        # forecasted wind, PV generation, etc.
        train_futureinputs = scaled_futureinputs[:-int(validationdays * 24), :]
        train_futureinputs = scaled_futureinputs[:-int(validationdays * 24), :].reshape(train_futureinputs.shape[0],
                                                                                        n_outsteps, featuredimension)
        train_label = scaled_label[:-int(validationdays * 24), :]
        train_label = train_label.reshape(train_label.shape[0], n_outsteps, 1)
        l_val = [int(24 * i - 1) for i in range(1, validationdays + 1)]
        validation_pastinputs = scaled_pastinputs[-int(validationdays * 24):, :]
        validation_pastinputs = validation_pastinputs[l_val, :]
        validation_pastinputs = validation_pastinputs.reshape(validation_pastinputs.shape[0], n_insteps,
                                                              featuredimension + len(label))
        validation_futureinputs = scaled_futureinputs[-int(validationdays * 24):, :]
        validation_futureinputs = validation_futureinputs[l_val, :]
        validation_futureinputs = validation_futureinputs.reshape(validation_futureinputs.shape[0], n_outsteps,
                                                                  featuredimension)
        validation_label = scaled_label[-int(validationdays * 24):, :]
        validation_label = validation_label[l_val, :]
        validation_label = validation_label.reshape(validation_label.shape[0], n_outsteps, 1)
        test_data = totaldata[-int(testdays * 24):, :]
        l_test = [int(24 * i - 1) for i in range(1, testdays + 1)]
        test_data = test_data[l_test, :]
        test_pastinputs = scaler_pastinputs.transform(test_data[:, :(featuredimension + len(label)) * n_insteps])
        test_pastinputs = test_pastinputs.reshape(test_data.shape[0], n_insteps, featuredimension + len(label))
        test_futureinputs = scaler_futureinputs.transform(
            test_data[:, (featuredimension + len(label)) * n_insteps: (featuredimension + len(label)) *
                                                                      n_insteps + featuredimension * n_outsteps])
        test_futureinputs = test_futureinputs.reshape(test_data.shape[0], n_outsteps, featuredimension)
        test_label = test_data[:, -n_outsteps:]
        test_label = test_label.reshape(test_data.shape[0], n_outsteps, 1)
    else:
        scaled_pastinputs = scaler_pastinputs.fit_transform(
            totaldata[:-testdays, :(featuredimension + len(label)) * n_insteps])
        scaled_futureinputs = scaler_futureinputs.fit_transform(
            totaldata[:-testdays,
            (featuredimension + len(label)) * n_insteps: (featuredimension + len(label)) *
                                                         n_insteps + featuredimension * n_outsteps])
        scaled_label = scaler_label.fit_transform(totaldata[:-testdays, -n_outsteps:])
        train_pastinputs = scaled_pastinputs[:-validationdays, :]
        # 3D [samples, timesteps, features]
        train_pastinputs = train_pastinputs.reshape(train_pastinputs.shape[0], n_insteps, featuredimension + len(label))
        train_futureinputs = scaled_futureinputs[:-validationdays, :]
        train_futureinputs = scaled_futureinputs[:-validationdays, :].reshape(train_futureinputs.shape[0],
                                                                              n_outsteps, featuredimension)
        train_label = scaled_label[:-validationdays, :]
        train_label = train_label.reshape(train_label.shape[0], n_outsteps, 1)
        validation_pastinputs = scaled_pastinputs[-validationdays:, :]
        validation_pastinputs = validation_pastinputs.reshape(validation_pastinputs.shape[0], n_insteps,
                                                              featuredimension + len(label))
        validation_futureinputs = scaled_futureinputs[-validationdays:, :]
        validation_futureinputs = validation_futureinputs.reshape(validation_futureinputs.shape[0], n_outsteps,
                                                                  featuredimension)
        validation_label = scaled_label[-validationdays:, :]
        validation_label = validation_label.reshape(validation_label.shape[0], n_outsteps, 1)
        test_data = totaldata[-testdays:, :]
        test_pastinputs = scaler_pastinputs.transform(test_data[:, :(featuredimension + len(label)) * n_insteps])
        test_pastinputs = test_pastinputs.reshape(test_data.shape[0], n_insteps, featuredimension + len(label))
        test_futureinputs = scaler_futureinputs.transform(
            test_data[:, (featuredimension + len(label)) * n_insteps: (featuredimension + len(label)) *
                                                                      n_insteps + featuredimension * n_outsteps])
        test_futureinputs = test_futureinputs.reshape(test_data.shape[0], n_outsteps, featuredimension)
        test_label = test_data[:, -n_outsteps:]
        test_label = test_label.reshape(test_data.shape[0], n_outsteps, 1)

    y_hatlist = []
    for i in range(runtimes):
        past_inputs = tf.keras.Input(shape=(n_insteps, featuredimension + 1), name='past_inputs')
        # Encoding the past
        encoder = tf.keras.layers.LSTM(hiddenunits, return_state=True)
        encoder_outputs, state_h, state_c = encoder(past_inputs)

        future_inputs = tf.keras.Input(shape=(n_outsteps, featuredimension), name='future_inputs')
        # Combining future inputs with recurrent output
        decoder_lstm = tf.keras.layers.LSTM(hiddenunits, return_sequences=True)
        x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])


        output = tf.keras.layers.Dense(1, activation=activation)(x)

        model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(loss='mae', optimizer=optimizer, metrics=["mae"])
        # print(model.summary())
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience,
                                              restore_best_weights=True)
        model.fit((train_pastinputs, train_futureinputs), train_label, epochs=epochs, batch_size=batch_size,
                  validation_data=((validation_pastinputs, validation_futureinputs), validation_label),
                  callbacks=[es], verbose=verbose)
        yhat = model.predict((test_pastinputs, test_futureinputs)).reshape(testdays, n_outsteps)
        yhat = scaler_label.inverse_transform(yhat)
        y_hatlist.append(yhat)

    yhat = np.asarray(y_hatlist).mean(axis=0)
    y = test_label.reshape(testdays, n_outsteps)
    return yhat, y


def run(daylist):
    yhatlist = []
    ylist = []
    timestamplist = []
    sectionbegin = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=daylist[0])
    sectionbegin = sectionbegin.strftime('%Y-%m-%d')
    if testdays == 1:
        for i in tqdm(daylist):
            predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i)
            timestamps, datacombined = collectingdata(inputdf=inputdf, n_insteps=n_insteps,
                                                      n_outsteps=n_outsteps,
                                                      predictiondate=predictiondate, lookbackyears=lookbackyears,
                                                      trainingdayscurrent=trainingdayscurrent,
                                                      trainingdaysprevious=trainingdaysprevious,
                                                      testdays=testdays)

            reframed = createDataset(datacombined, label=label, features=features, n_insteps=n_insteps,
                                     n_outsteps=n_outsteps, daydiscretization=daydiscretization)
            reframedvalues = reframed.values
            yhat, y = trainandpredict(totaldata=reframedvalues, scaler=scaler, n_insteps=n_insteps,
                                      n_outsteps=n_outsteps, featuredimension=featuredimension, label=label,
                                      daydiscretization=daydiscretization, validationdays=validationdays,
                                      testdays=testdays, hiddenunits=hiddenunits, activation=activation,
                                      epochs=epochs, batch_size=batchsize,
                                      patience=patience, runtimes=runtimes, verbose=verbose)
            timestamplist.append(timestamps)
            yhatlist.append(yhat)
            ylist.append(y)
    else:
        for i in tqdm(range(testdays - 1, daystopredict, testdays)):
            predictiondateend = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i)
            timestamps, datacombined = collectingdata(inputdf=inputdf, n_insteps=n_insteps,
                                                      n_outsteps=n_outsteps,
                                                      predictiondate=predictiondateend,
                                                      lookbackyears=lookbackyears,
                                                      trainingdayscurrent=trainingdayscurrent,
                                                      trainingdaysprevious=trainingdaysprevious,
                                                      testdays=testdays)

            reframed = createDataset(datacombined, label=label, features=features, n_insteps=n_insteps,
                                     n_outsteps=n_outsteps, daydiscretization=daydiscretization)
            reframedvalues = reframed.values
            yhat, y = trainandpredict(totaldata=reframedvalues, scaler=scaler, n_insteps=n_insteps,
                                      n_outsteps=n_outsteps, featuredimension=featuredimension, label=label,
                                      daydiscretization=daydiscretization, validationdays=validationdays,
                                      testdays=testdays, hiddenunits=hiddenunits,
                                      activation=activation, epochs=epochs, batch_size=batchsize,
                                      patience=patience, runtimes=runtimes, verbose=verbose)
            timestamplist.append(timestamps)
            yhatlist.append(yhat)
            ylist.append(y)

    predictionlist = np.asarray(yhatlist).flatten()
    actuallist = np.asarray(ylist).flatten()
    timestamplist = np.asarray(timestamplist).flatten()
    error = actuallist - predictionlist
    predictiondf = pd.DataFrame(
        {'timestamp': timestamplist, 'prediction': predictionlist, 'actual': actuallist, 'error': error})
    filename = sectionbegin + remarks
    filepath = path.join('Prediction/EncoderDecoder', filename + '.csv')
    predictiondf.to_csv(filepath, sep=';', index=False)


if __name__ == '__main__':
    with open('Config/config_encoderdecoder.json', 'r') as fp:
        parameterdict = json.load(fp)
    inputfilepath = parameterdict['inputfilepath']
    inputdf = pd.read_csv(inputfilepath, sep=';', index_col=['UTC Timestamp'])
    label = parameterdict['label']
    features = parameterdict['features']
    featuredimension = len(features)
    allcolumns = label + features
    inputdf = inputdf[allcolumns]
    predictionbegin = parameterdict['predictionbegin']
    daystopredict = parameterdict['daystopredict']
    validationdays = parameterdict['validationdays']
    testdays = parameterdict['testdays']
    runtimes = parameterdict['runtimes']
    verbose = parameterdict['verbose']

    # hyperparameters for the deep neural networks
    hyperparameters = parameterdict['hyperparameters']
    params_grid = ParameterGrid(hyperparameters)

    with open('Data/sampledays2018.json', 'r') as fp:
        daylist = json.load(fp)
    daydeltalist = [
        int((datetime.strptime(item, "%Y-%m-%d") - datetime.strptime(predictionbegin,
                                                                     "%Y-%m-%d")).total_seconds() / 86400)
        for item in daylist]
    for enum, params in enumerate(tqdm(params_grid)):
        start_time = time.time()
        daydiscretization = params['daydiscretization']
        lookbackyears = params['lookbackyears']
        trainingdayscurrent = params['trainingdayscurrent']
        trainingdaysprevious = params['trainingdaysprevious']
        n_insteps = params['n_insteps']
        n_outsteps = params['n_outsteps']
        scaler = params['scaler']
        hiddenunits = params['hiddenunits']
        activation = params['activation']
        patience = params['patience']
        epochs = params['epochs']
        batchsize = params['batchsize']
        remarks = 'EnDe' + 'LB' + str(lookbackyears) + 'VD' + str(validationdays) + 'Dis' + str(
            daydiscretization) + 'Hidden' + str(hiddenunits) + 'Steps' + str(
            n_insteps)
        run(list(range(daystopredict)))
