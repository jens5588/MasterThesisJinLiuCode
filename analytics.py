import numpy as np
import pandas as pd
from datetime import datetime
import json
import random


def generatedays(inputfilepath, days):
    # generate 60 sample days for year 2018
    df = pd.read_csv('Prediction/DirectMLP/2018DirectMLPLB3VD5hiddenL4S64.csv', sep=';')
    monthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    weekdayfilled = True
    while (weekdayfilled):
        totallist = []
        weekdaydict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        # for 12 months
        for i in range(1, 13):
            # total data in each month
            monthdf = df.loc[sum(monthdays[:i - 1]) * 24:sum(monthdays[:i]) * 24 - 1, :]
            actual = np.mean(np.absolute(monthdf.actual))
            datelist = monthdf['timestamp'].to_list()
            datelist = [datelist[item] for item in range(0, len(datelist), 24)]
            datelist = [item.split(' ')[0] for item in datelist]
            sampleactual = 0
            monthdf = monthdf.set_index('timestamp')
            # the difference between mae of 5 sample days and mae of the whole month should not exceed 0.05
            while (not actual - 0.05 < sampleactual < actual + 0.05):
                i = 0
                l = []
                # 5 days for each month
                while (i < 5):
                    num = random.randint(0, len(datelist) - 1)
                    datum = datelist[num]
                    if datum not in l:
                        l.append(datum)
                        i += 1
                l = sorted(l)
                tmp = monthdf.loc[l[0] + ' 00':l[0] + ' 23', :]
                for item in l[1:]:
                    tmp1 = monthdf.loc[item + ' 00':item + ' 23', :]
                    tmp = pd.concat([tmp, tmp1], ignore_index=True)
                sampleactual = np.mean(np.absolute(tmp.actual))

            totallist.append(l)
        totallist = sum(totallist, [])
        for item in totallist:
            weekday = datetime.strptime(item, "%Y-%m-%d").isoweekday()
            weekdaydict[str(weekday)] += 1
        # weekday criterion the number of each weekday in 60 sample days should be between 6 and 10. So balance
        # of weekdays in the sample days
        if all(int(5 * 12 / 7) - 2 < value < int(5 * 12 / 7) + 2 for value in weekdaydict.values()):
            weekdayfilled = False
        print(weekdaydict)

    with open('Data/sampledays2018.json', 'w') as fp:
        json.dump(totallist, fp)

def daily_error_analysis(df, predictionmethods):
    datelist = []
    mae_actual_list = []
    mae_error_list = []
    for i in range(int(df.shape[0] / 24)):
        begin = int(24 * i)
        end = begin + 23
        date = df.loc[begin, 'timestamp'].split(' ')[0]
        mae_error = np.mean(np.absolute(df.loc[begin:end, 'error'].values))
        std_error = np.std(df.loc[begin:end, 'error'])
        mae_actual = np.mean(np.absolute(df.loc[begin:end, 'actual'].values))
        std_actual = np.std(df.loc[begin:end, 'error'])
        datelist.append(date)
        mae_actual_list.append(mae_actual)
        mae_error_list.append(mae_error)
    weekdaylist = [datetime.strptime(item, "%Y-%m-%d").isoweekday() for item in datelist]
    with open('holidays.json', 'r') as holidayfile:
        data = holidayfile.read()
    holidays = json.loads(data)
    totalholidayslist = [*holidays["2010"]] + [*holidays["2011"]] + [*holidays["2012"]] + [*holidays["2013"]] + \
                        [*holidays["2014"]] + [*holidays["2015"]] + [*holidays["2016"]] + [*holidays["2017"]] + [
                            *holidays["2018"]]
    holidayslist = [1 if item in totalholidayslist else 0 for item in datelist]
    analytics_df = pd.DataFrame({'timestamp': datelist, 'weekday': weekdaylist, 'holiday': holidayslist,
                                 'prediction mae': mae_error_list, 'actual mae': mae_actual_list})
    filename = 'analytics' + predictionmethods + '.csv'
    analytics_df.to_csv(filename, sep=';', index=False)


def benchdaysAnalysis(inputfile, dayfile):
    with open(dayfile, 'r') as fp:
        daylist = json.load(fp)
    df = pd.read_csv(inputfile, sep=';', index_col=['timestamp'])
    tmp = df.loc[daylist[0] + ' 00':daylist[0] + ' 23', :]
    for day in daylist[1:]:
        tmp1 = df.loc[day + ' 00':day + ' 23', :]
        tmp = pd.concat([tmp, tmp1], ignore_index=True)
    actual = np.round(np.mean(np.absolute(tmp.actual)), decimals=3)
    error = np.round(np.mean(np.absolute(tmp.error)), decimals=3)

    return actual, error


def maeandrmseyearly(inputfilepath, decimals=3):
    df = pd.read_csv(inputfilepath, sep=';', index_col=['UTC Timestamp'])
    yearlist = ['2013', '2014', '2015', '2016', '2017', '2018']
    for year in yearlist:
        tmp = df.loc[year + '-01-01 00':year + '-12-31 23', :]
        fundamental_mae = np.round(np.mean(np.absolute(tmp['price difference'])), decimals=decimals)
        fundamental_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['price difference']) ** 2)), decimals=decimals)
        directmlp_mae = np.round(np.mean(np.absolute(tmp['Error Direct MLP'])), decimals=decimals)
        directmlp_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Direct MLP']) ** 2)), decimals=decimals)
        sarimax_mae = np.round(np.mean(np.absolute(tmp['Error SARIMAX'])), decimals=decimals)
        sarimax_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error SARIMAX']) ** 2)), decimals=decimals)
        recursivemlp_mae = np.round(np.mean(np.absolute(tmp['Error Recursive MLP'])), decimals=decimals)
        recursivemlp_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Recursive MLP']) ** 2)), decimals=decimals)
        ende_mae = np.round(np.mean(np.absolute(tmp['Error Encoder-Decoder'])), decimals=decimals)
        ende_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Encoder-Decoder']) ** 2)), decimals=decimals)
        knn_mae = np.round(np.mean(np.absolute(tmp['Error KNN'])), decimals=decimals)
        knn_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error KNN']) ** 2)), decimals=decimals)
        naive_mae = np.round(np.mean(np.absolute(tmp['Error Naive'])), decimals=decimals)
        naive_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Naive']) ** 2)), decimals=decimals)

        print(year)
        print('-------------------------')
        print('Fundamental MAE: ' + str(fundamental_mae))
        print('Fundamental RMSE: ' + str(fundamental_rmse))
        print('Direct MLP MAE: ' + str(directmlp_mae))
        print('Direct MLP RMSE: ' + str(directmlp_rmse))
        print('Sarimax MAE: ' + str(sarimax_mae))
        print('Sarimax RMSE: ' + str(sarimax_rmse))
        print('Recursive MLP MAE: ' + str(recursivemlp_mae))
        print('Recursive MLP RMSE: ' + str(recursivemlp_rmse))
        print('Encoder-Decoder MAE: ' + str(ende_mae))
        print('Encoder-Decoder RMSE: ' + str(ende_rmse))
        print('KNN MAE: ' + str(knn_mae))
        print('KNN RMSE: ' + str(knn_rmse))
        print('Naive MAE: ' + str(naive_mae))
        print('Naive RMSE: ' + str(naive_rmse))


def maeandrmsehourly(inputfilepath, decimals=2):
    df = pd.read_csv(inputfilepath, sep=';', index_col=['UTC Timestamp'])
    df = df.loc['2018-01-01 00':'2018-12-31 23', :]
    for hour in range(24):
        tmp = df[df.Hour == hour]
        fundamental_mae = np.round(np.mean(np.absolute(tmp['price difference'])), decimals=decimals)
        fundamental_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['price difference']) ** 2)), decimals=decimals)
        directmlp_mae = np.round(np.mean(np.absolute(tmp['Error Direct MLP'])), decimals=decimals)
        directmlp_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Direct MLP']) ** 2)), decimals=decimals)
        sarimax_mae = np.round(np.mean(np.absolute(tmp['Error SARIMAX'])), decimals=decimals)
        sarimax_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error SARIMAX']) ** 2)), decimals=decimals)
        recursivemlp_mae = np.round(np.mean(np.absolute(tmp['Error Recursive MLP'])), decimals=decimals)
        recursivemlp_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Recursive MLP']) ** 2)), decimals=decimals)
        ende_mae = np.round(np.mean(np.absolute(tmp['Error Encoder-Decoder'])), decimals=decimals)
        ende_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Encoder-Decoder']) ** 2)), decimals=decimals)
        knn_mae = np.round(np.mean(np.absolute(tmp['Error KNN'])), decimals=decimals)
        knn_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error KNN']) ** 2)), decimals=decimals)
        naive_mae = np.round(np.mean(np.absolute(tmp['Error Naive'])), decimals=decimals)
        naive_rmse = np.round(np.sqrt(np.mean(np.asarray(tmp['Error Naive']) ** 2)), decimals=decimals)

        print('Hour: ' + str(hour))
        print('-------------------------')
        print('Fundamental MAE: ' + str(fundamental_mae))
        print('Fundamental RMSE: ' + str(fundamental_rmse))
        print('Direct MLP MAE: ' + str(directmlp_mae))
        print('Direct MLP RMSE: ' + str(directmlp_rmse))
        print('Sarimax MAE: ' + str(sarimax_mae))
        print('Sarimax RMSE: ' + str(sarimax_rmse))
        print('Recursive MLP MAE: ' + str(recursivemlp_mae))
        print('Recursive MLP RMSE: ' + str(recursivemlp_rmse))
        print('Encoder-Decoder MAE: ' + str(ende_mae))
        print('Encoder-Decoder RMSE: ' + str(ende_rmse))
        print('KNN MAE: ' + str(knn_mae))
        print('KNN RMSE: ' + str(knn_rmse))
        print('Naive MAE: ' + str(naive_mae))
        print('Naive RMSE: ' + str(naive_rmse))


if __name__ == '__main__':
    # maeandrmsehourly('Data/2013-2018Label.csv')
    maeandrmseyearly('Data/2013-2018Label.csv', 2)
