import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import operator
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# between two models, Direct MLP, SARIMAX labelling the best model
def labelling(a, b):
    return np.argmin(np.absolute(np.asarray([a, b])))

# among two models, Direct MLP, SARIMAX, recursive MLP, labelling the best model
def labelling3(a, b, c):
    return np.argmin(np.absolute(np.asarray([a, b, c])))

# among four models, Direct MLP, SARIMAX, recursive MLP, encoder-decoder model, labelling the best model
def labelling4(a, b, c, d):
    return np.argmin(np.absolute(np.asarray([a, b, c, d])))


if __name__ == '__main__':
    inputdf = pd.read_csv("Data/preprocessed price.csv", sep=';', index_col=['UTC Timestamp'])
    inputdf = inputdf.loc['2013-01-01 00':'2018-12-31 23', :]

    directmlp2013 = pd.read_csv('Prediction/DirectMLP/2013DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2013 = pd.read_csv('Prediction/SARIMAX/2013SARIMAXAICpQ.csv', sep=';')
    recursivemlp2013 = pd.read_csv('Prediction/RecursiveMLP/2013RecursiveMLPLB3VD5Lag6hiddenL4S64.csv', sep=';')
    ende2013 = pd.read_csv('Prediction/EncoderDecoder/2013EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2013 = pd.read_csv('Prediction/KNN/2013KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')

    directmlp2014 = pd.read_csv('Prediction/DirectMLP/2014DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2014 = pd.read_csv('Prediction/SARIMAX/2014SARIMAXAICpQ.csv', sep=';')
    recursivemlp2014 = pd.read_csv('Prediction/RecursiveMLP/2014RecursiveMLPLB3VD5Lag6hiddenL4S64.csv', sep=';')
    ende2014 = pd.read_csv('Prediction/EncoderDecoder/2014EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2014 = pd.read_csv('Prediction/KNN/2014KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')

    directmlp2015 = pd.read_csv('Prediction/DirectMLP/2015DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2015 = pd.read_csv('Prediction/SARIMAX/2015SARIMAXAICpQ.csv', sep=';')
    recursivemlp2015 = pd.read_csv('Prediction/RecursiveMLP/2015RecursiveMLPLB3VD5Lag6hiddenL4S64.csv', sep=';')
    ende2015 = pd.read_csv('Prediction/EncoderDecoder/2015EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2015 = pd.read_csv('Prediction/KNN/2015KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')

    directmlp2016 = pd.read_csv('Prediction/DirectMLP/2016DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2016 = pd.read_csv('Prediction/SARIMAX/2016SARIMAXAICpQ.csv', sep=';')
    recursivemlp2016 = pd.read_csv('Prediction/RecursiveMLP/2016RecursiveMLPLB3VD5Lag6hiddenL4S64.csv', sep=';')
    ende2016 = pd.read_csv('Prediction/EncoderDecoder/2016EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2016 = pd.read_csv('Prediction/KNN/2016KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')

    directmlp2017 = pd.read_csv('Prediction/DirectMLP/2017DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2017 = pd.read_csv('Prediction/SARIMAX/2017SARIMAXAICpQ.csv', sep=';')
    recursivemlp2017 = pd.read_csv('Prediction/RecursiveMLP/2017RecursiveMLPLB3VD5Lag6hiddenL4S64.csv', sep=';')
    ende2017 = pd.read_csv('Prediction/EncoderDecoder/2017EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2017 = pd.read_csv('Prediction/KNN/2017KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')

    directmlp2018 = pd.read_csv('Prediction/DirectMLP/2018DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    sarimax2018 = pd.read_csv('Prediction/SARIMAX/2018SARIMAXAICpQ.csv', sep=';')
    recursivemlp2018 = pd.read_csv(
        'Prediction/RecursiveMLP/2018RecursiveMLPLB3VD5Lag6hiddenL4S64.csv',
        sep=';')
    ende2018 = pd.read_csv('Prediction/EncoderDecoder/2018EnDeLB3VD5Hidden64Steps24.csv', sep=';')
    knn2018 = pd.read_csv('Prediction/KNN/2018KNNLB3Neighbours10MinMaxVariableWeighted.csv', sep=';')
    inputdf[
        'Prediction Direct MLP'] = directmlp2013.prediction.tolist() + directmlp2014.prediction.tolist() + \
                                   directmlp2015.prediction.tolist() + directmlp2016.prediction.tolist() + \
                                   directmlp2017.prediction.tolist() + directmlp2018.prediction.tolist()

    inputdf[
        'Prediction SARIMAX'] = sarimax2013.prediction.tolist() + sarimax2014.prediction.tolist() + \
                                sarimax2015.prediction.tolist() + sarimax2016.prediction.tolist() + \
                                sarimax2017.prediction.tolist() + sarimax2018.prediction.tolist()
    inputdf[
        'Prediction Recursive MLP'] = recursivemlp2013.prediction.tolist() + recursivemlp2014.prediction.tolist() \
                                      + recursivemlp2015.prediction.tolist() + recursivemlp2016.prediction.tolist() \
                                      + recursivemlp2017.prediction.tolist() + recursivemlp2018.prediction.tolist()
    inputdf[
        'Prediction Encoder-Decoder'] = ende2013.prediction.tolist() + ende2014.prediction.tolist() + \
                                        ende2015.prediction.tolist() + ende2016.prediction.tolist() + \
                                        ende2017.prediction.tolist() + ende2018.prediction.tolist()
    inputdf[
        'Prediction KNN'] = knn2013.prediction.tolist() + knn2014.prediction.tolist() + \
                            knn2015.prediction.tolist() + knn2016.prediction.tolist() + \
                            knn2017.prediction.tolist() + knn2018.prediction.tolist()
# Calculating residual forecasting for the naive model
inputdf['Prediction Naive'] = 0.5 * inputdf['price difference t_24'] + 0.5 * inputdf['price difference t_168']
# Calculating residual forecasting for ensemble model from two best models
inputdf['Prediction Average'] = 0.5 * inputdf['Prediction Direct MLP'] + 0.5 * inputdf['Prediction SARIMAX']
# Calculating residual forecasting for ensemble model from two best models
inputdf['Prediction Average3'] = 1 / 3 * inputdf['Prediction Direct MLP'] + 1 / 3 * inputdf[
    'Prediction SARIMAX'] + 1 / 3 * inputdf['Prediction Recursive MLP']
# Calculating residual forecasting for ensemble model from four best models
inputdf['Prediction Average4'] = 1 / 4 * inputdf['Prediction Direct MLP'] + 1 / 4 * inputdf[
    'Prediction SARIMAX'] + 1 / 4 * inputdf['Prediction Recursive MLP'] + 1 / 4 * inputdf['Prediction Encoder-Decoder']

# Residual forecasting errors, 'price difference' is the same for residual
inputdf['Error Direct MLP'] = inputdf['price difference'] - inputdf['Prediction Direct MLP']
inputdf['Error SARIMAX'] = inputdf['price difference'] - inputdf['Prediction SARIMAX']
inputdf['Error Recursive MLP'] = inputdf['price difference'] - inputdf['Prediction Recursive MLP']
inputdf['Error Encoder-Decoder'] = inputdf['price difference'] - inputdf['Prediction Encoder-Decoder']
inputdf['Error KNN'] = inputdf['price difference'] - inputdf['Prediction KNN']
inputdf['Error Naive'] = inputdf['price difference'] - inputdf['Prediction Naive']
inputdf['Error Average'] = inputdf['price difference'] - inputdf['Prediction Average']
inputdf['Error Average3'] = inputdf['price difference'] - inputdf['Prediction Average3']
inputdf['Error Average4'] = inputdf['price difference'] - inputdf['Prediction Average4']

# final submitted price, price forecasting of the fundamental model + residual forecasting
inputdf['Price Direct MLP'] = inputdf['price estimator Model'] + inputdf['Prediction Direct MLP']
inputdf['Price SARIMAX'] = inputdf['price estimator Model'] + inputdf['Prediction SARIMAX']
inputdf['Price Recursive MLP'] = inputdf['price estimator Model'] + inputdf['Prediction Recursive MLP']
inputdf['Price Encoder-Decoder'] = inputdf['price estimator Model'] + inputdf['Prediction Encoder-Decoder']
inputdf['Price KNN'] = inputdf['price estimator Model'] + inputdf['Prediction KNN']
inputdf['Price Naive'] = inputdf['price estimator Model'] + inputdf['Prediction Naive']
inputdf['Price Average'] = inputdf['price estimator Model'] + inputdf['Prediction Average']
inputdf['Price Average3'] = inputdf['price estimator Model'] + inputdf['Prediction Average3']
inputdf['Price Average4'] = inputdf['price estimator Model'] + inputdf['Prediction Average4']

inputdf['UTC Timestamp'] = inputdf.index.tolist()
inputdf = inputdf.reset_index(drop=True)

inputdf['Label'] = inputdf.apply(lambda x: labelling(x['Error Direct MLP'], x['Error SARIMAX']), axis=1)
inputdf['Label3'] = inputdf.apply(
    lambda x: labelling3(x['Error Direct MLP'], x['Error SARIMAX'], x['Error Recursive MLP']), axis=1)
inputdf['Label4'] = inputdf.apply(
    lambda x: labelling4(x['Error Direct MLP'], x['Error SARIMAX'], x['Error Recursive MLP'],
                         x['Error Encoder-Decoder']), axis=1)
inputdf.to_csv('Data/2013-2018Label.csv', sep=';', index=False)
