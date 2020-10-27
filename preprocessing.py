import pandas as pd
import json
import random
from datetime import datetime
from datetime import timedelta
import plotly.offline as offline
import plotly.graph_objs as go


def preprocessing(inputfilepath='Data/Table_Fundamentalmodell_InOut.xlsx',
                  outputfilepath='Data/preprocessed price.csv'):
    df = pd.read_excel(inputfilepath, parse_dates=['UTC Timestamp'])
    df = df.drop(['Unnamed: 0'], axis=1)
    df['price difference'] = df['price electricity real'] - df['price estimator Model']
    df['UTC Timestamp'] = df['UTC Timestamp'].apply(lambda x: (x.to_pydatetime() + timedelta(minutes=1)).strftime(
        "%Y-%m-%d %H"))
    df['Hour'] = df['UTC Timestamp'].apply(lambda x: int(x.split(" ")[1]))
    df['Demand t_1'] = df['Demand'].shift(1)
    df['DemandChange'] = df['Demand'] - df['Demand t_1']
    df['Weekday'] = df['UTC Timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H").isoweekday())
    df['Year'] = df['UTC Timestamp'].apply(lambda x: int(x.split(' ')[0].split('-')[0]))
    df['Month'] = df['UTC Timestamp'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    # 1 for spring, 2 for summer, 3 for autumn and 4 for winter
    seasonlist = [4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    df['Season'] = df['Month'].apply(lambda x: seasonlist[x - 1])

    with open('Data/holidays.json', 'r') as holidayfile:
        data = holidayfile.read()
    holidays = json.loads(data)
    holidayslist = [*holidays["2010"]] + [*holidays["2011"]] + [*holidays["2012"]] + [*holidays["2013"]] + \
                   [*holidays["2014"]] + [*holidays["2015"]] + [*holidays["2016"]] + [*holidays["2017"]] + [
                       *holidays["2018"]]
    df['Holiday'] = df['UTC Timestamp'].apply(lambda x: 1 if x.split(" ")[0] in holidayslist else 0)
    working_weekdayl = df['Weekday'].to_list()
    working_weekdayl = [1 if working_weekdayl[i] < 6 else 0 for i in range(len(working_weekdayl))]
    working_holidayl = df['Holiday'].to_list()
    working_holidayl = [int(not bool(working_holidayl[i])) for i in range(len(working_holidayl))]
    workingdaylist = [working_holidayl[i] * working_weekdayl[i] for i in range(len(working_weekdayl))]
    df['Working day'] = workingdaylist
    df['NTC NET PL'] = df['NTC from PL'] - df['NTC to PL']
    df['NTC NET CZ'] = df['NTC from CZ'] - df['NTC to CZ']
    df['NTC NET CH'] = df['NTC from CH'] - df['NTC to CH']
    df['NTC NET FR'] = df['NTC from FR'] - df['NTC to FR']
    df['NTC NET NL'] = df['NTC from NL'] - df['NTC to NL']
    df['NTC NET DK-1'] = df['NTC from DK-1'] - df['NTC to DK-1']
    df['NTC NET DK-2'] = df['NTC from DK-2'] - df['NTC to DK-2']
    df['NTC NET SE-4'] = df['NTC from SE-4'] - df['NTC to SE-4']
    df['NTC NET'] = df['NTC NET PL'] + df['NTC NET CZ'] + df['NTC NET CH'] + df['NTC NET FR'] + df['NTC NET NL'] + df[
        'NTC NET DK-1'] + df['NTC NET DK-2'] + df['NTC NET SE-4']
    df['NTC Import'] = df['NTC from PL'] + df['NTC from CZ'] + df['NTC from CH'] + df['NTC from FR'] + df[
        'NTC from NL'] + \
                       df['NTC from DK-1'] + df['NTC from DK-2'] + df['NTC from SE-4']
    df['NTC Export'] = df['NTC to PL'] + df['NTC to CZ'] + df['NTC to CH'] + df['NTC to FR'] + df['NTC to NL'] + \
                       df['NTC to DK-1'] + df['NTC to DK-2'] + df['NTC to SE-4']
    df['Outages Net'] = df['Outages Nuclear'] + df['Outages Lignite'] + df['Outages Hard Coal'] + df['Outages Gas']
    df = df.drop(
        ['Demand t_1', 'NTC from PL', 'NTC to PL', 'NTC from CZ', 'NTC to CZ', 'NTC from CH', 'NTC to CH',
         'NTC from FR',
         'NTC to FR', 'NTC from NL', 'NTC to NL', 'NTC from DK-1', 'NTC to DK-1', 'NTC from DK-2', 'NTC to DK-2',
         'NTC from SE-4', 'NTC to SE-4', 'NTC to AT', 'NTC from AT', 'NTC NET PL', 'NTC NET CZ', 'NTC NET CH',
         'NTC NET FR',
         'NTC NET NL', 'NTC NET DK-1', 'NTC NET DK-2', 'NTC NET SE-4', 'cap_max Reservoir AT', 'cap_max Reservoir CH',
         'availability Run-of-River DE'], axis=1)
    lastindex = df[df['PV Prediction'].isna()].index.to_list()[-1]
    for i in range(lastindex + 1):
        df.loc[i, 'PV Prediction'] = df.loc[i, 'PV']
        df.loc[i, 'Wind Prediction'] = df.loc[i, 'Wind']
    df['RenewableEnergy'] = df['PV Prediction'] + df['Wind Prediction']
    laggedcolumns = ['price estimator Model', 'price difference', 'price electricity real']
    for column in laggedcolumns:
        for i in range(1, 505):
            df[column + ' t_' + str(i)] = df[column].shift(i)
    df.to_csv(outputfilepath, sep=';', index=False)

if __name__ == '__main__':
    preprocessing('Data/Table_Fundamentalmodell_InOut.xlsx','Data/preprocessed price.csv')



