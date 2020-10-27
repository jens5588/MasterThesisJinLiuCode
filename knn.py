import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm
from os import path


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
        datacombined = pd.concat([combinedprevious, datacurrent], ignore_index=True).dropna().reset_index(drop=True)
    else:
        datacombined = datacurrent.dropna().reset_index(drop=True)
    return timestamps, datacombined


def kNearestNeighbour(df, k):
    rowindexlist = df.index.tolist()[:-1]
    usedfeatures = features[1:]
    featureweightslist = [np.square(df.loc[rowindexlist, 'price difference'].corr(df.loc[rowindexlist, feature])) for
                          feature in usedfeatures]
    featureweightslist = [0 if np.isnan(weight) else weight for weight in featureweightslist]
    normedfeatureweightslist = [weight / sum(featureweightslist) for weight in featureweightslist]
    scaler = MinMaxScaler()
    ylist = df['price difference'].values
    x = df.values[:, 2:]
    scaled_x = scaler.fit_transform(x)
    objective = scaled_x[-1]
    difference = scaled_x[:-1] - objective
    distancelist = np.sqrt(np.square(difference).dot(np.asarray(normedfeatureweightslist)))
    # distancelist = np.sqrt(np.sum(np.square(difference), axis=1))
    neighboursindexlist = list(np.argsort(distancelist)[:k])
    neighboursdistancelist = [distancelist[index] for index in neighboursindexlist]
    neighboursweightslist = [1 / neighbourdistance for neighbourdistance in neighboursdistancelist]
    normedweightslist = [weight / sum(neighboursweightslist) for weight in neighboursweightslist]
    neighboursvalues = [ylist[index] for index in neighboursindexlist]

    yhat = np.dot(np.asarray(neighboursvalues), np.asarray(normedweightslist))

    y = ylist[-1]

    return yhat, y


def run(daylist):
    yhatlist = []
    ylist = []
    timestamplist = []
    sectionbegin = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=daylist[0])
    sectionbegin = sectionbegin.strftime('%Y-%m-%d')
    for i in tqdm(daylist):
        predictiondate = datetime.strptime(predictionbegin, '%Y-%m-%d') + timedelta(days=i)
        timestamps, datacombined = collectingdata(inputdftmp, predictiondate, lookbackyears,
                                                  trainingdayscurrent, trainingdaysprevious, testdays)
        for hour in range(24):
            tmp = datacombined[datacombined.Hour == hour]
            yhat, y = kNearestNeighbour(tmp, neighbours)
            yhatlist.append(yhat)
            ylist.append(y)
        timestamplist.append(timestamps)

    predictionlist = np.asarray(yhatlist).flatten()
    actuallist = np.asarray(ylist).flatten()
    timestamplist = np.asarray(timestamplist).flatten()
    error = actuallist - predictionlist
    print('MAE: ' + str(np.mean(np.absolute((error)))))
    predictiondf = pd.DataFrame(
        {'timestamp': timestamplist, 'prediction': predictionlist, 'actual': actuallist, 'error': error})
    filename = sectionbegin + '_' + remarks
    filepath = path.join('Prediction/KNN', filename + '.' + 'csv')
    predictiondf.to_csv(filepath, sep=';', index=False)


if __name__ == '__main__':
    predictionbegin = '2018-01-01'
    daystopredict = 365
    lookbackyears = 3
    trainingdayscurrent = 49
    trainingdaysprevious = 49
    testdays = 1
    neighbours = 10
    inputdf = pd.read_csv('Data/preprocessed price.csv', sep=';', index_col=['UTC Timestamp'])
    features = ["Hour", "Weekday", "Holiday", "daily temperature", "price estimator Model", "Demand",
                "PV Prediction", "Wind Prediction", "price difference t_24", "price difference t_25",
                "price difference t_26", "price difference t_27", "price difference t_28",
                "price difference t_47", "price difference t_48", "price difference t_49",
                "price difference t_71", "price difference t_72", "price difference t_73",
                "price difference t_95", "price difference t_96", "price difference t_97",
                "price difference t_119", "price difference t_120", "price difference t_121",
                "price difference t_143", "price difference t_144", "price difference t_145",
                "price difference t_167", "price difference t_168", "price difference t_169", "price difference t_336",
                "price difference t_504", "NTC NET", "Price CO2", "Price Gas", "Price Hard Coal",
                "Price Oil"]
    allcolumns = ['price difference'] + features
    inputdftmp = inputdf[allcolumns]
    remarks = 'KNN' + 'LB' + str(lookbackyears) + 'Neighbours' + str(neighbours) + 'MinMax' + 'VariableWeighted'
    run(list(range(daystopredict)))
