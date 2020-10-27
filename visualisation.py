import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.offline as offline
import plotly.graph_objs as go


def linevisualisation(inputfilepath):
    df = pd.read_csv(inputfilepath, sep=';')
    df['date'] = df['timestamp'].apply(lambda x: x.split(" ")[0])
    df['hour'] = df['timestamp'].apply(lambda x: x.split(" ")[1])
    datelist = df['date'].unique().tolist()
    data = []
    for item in datelist:
        tmp = df[df.date == item]
        y = tmp['actual'].values.tolist()
        x = tmp['hour'].values.tolist()
        weekday = datetime.strptime(item, '%Y-%m-%d').date().isoweekday()
        if weekday < 6:
            trace = go.Scatter(x=x, y=y, name=item + '-Weekday-' + str(weekday))
            data.append(trace)
    layout = go.Layout(title='price difference prediction', xaxis={"title": "Hour"},
                       yaxis={"title": "Price difference [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'price_difference_2018_weekday.html'
    offline.plot(fig, filename=name)


def weeklyvisualisation(inputfilepath1, inputfilepath2, inputfilepath3, inputfilepath4):
    df1 = pd.read_csv(inputfilepath1, sep=';')
    df2 = pd.read_csv(inputfilepath2, sep=';')
    df3 = pd.read_csv(inputfilepath3, sep=';')
    df4 = pd.read_csv(inputfilepath4, sep=';')
    weeks = int(df1.shape[0] / 168)
    x = list(range(1, 169))
    data = []
    for i in range(weeks):
        weekbegin = df1.loc[int(i * 168), 'timestamp'].split(' ')[0]
        actual = df1.loc[int(i * 168):int(i * 168 + 167), 'actual']
        erroractual = np.round(np.mean(np.absolute(df1.loc[int(i * 168):int(i * 168 + 167), 'actual'])), decimals=3)
        prediction1 = df1.loc[int(i * 168):int(i * 168 + 167), 'prediction']
        error1 = np.round(np.mean(np.absolute(df1.loc[int(i * 168):int(i * 168 + 167), 'error'])), decimals=3)
        prediction2 = df2.loc[int(i * 168):int(i * 168 + 167), 'prediction']
        error2 = np.round(np.mean(np.absolute(df2.loc[int(i * 168):int(i * 168 + 167), 'error'])), decimals=3)
        prediction3 = df3.loc[int(i * 168):int(i * 168 + 167), 'prediction']
        error3 = np.round(np.mean(np.absolute(df3.loc[int(i * 168):int(i * 168 + 167), 'error'])), decimals=3)
        prediction4 = df4.loc[int(i * 168):int(i * 168 + 167), 'prediction']
        error4 = np.round(np.mean(np.absolute(df4.loc[int(i * 168):int(i * 168 + 167), 'error'])), decimals=3)
        traceactual = go.Scatter(x=x, y=actual, name=weekbegin + '- Actual: ' + str(erroractual))
        trace1 = go.Scatter(x=x, y=prediction1, name=weekbegin + '- Prediction1: ' + str(error1))
        trace2 = go.Scatter(x=x, y=prediction2, name=weekbegin + '- Prediction2: ' + str(error2))
        trace3 = go.Scatter(x=x, y=prediction3, name=weekbegin + '- Prediction3: ' + str(error3))
        trace4 = go.Scatter(x=x, y=prediction4, name=weekbegin + '- Prediction4: ' + str(error4))
        data.append(traceactual)
        data.append(trace1)
        data.append(trace2)
        data.append(trace3)
        data.append(trace4)
    layout = go.Layout(title='weekly prediction', xaxis={"title": "Hour"},
                       yaxis={"title": "Price difference [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'price_difference_2018_yearly.html'
    offline.plot(fig, filename=name)


def scattervisualisation(inputfilepath):
    df = pd.read_csv(inputfilepath, sep=';')
    data = []
    for i in range(1, 8):
        tmp = df[df.Weekday == i]
        trace = go.Scatter(x=tmp['Hour'].values.tolist(), y=tmp['price difference'].values.tolist(),
                           name='Weekday' + str(i),
                           mode='markers')
        data.append(trace)
    layout = go.Layout(title='price difference scatter plot', xaxis={"title": "Hour"},
                       yaxis={"title": "Price difference [€/MWh]"})

    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'price_difference_scatter_plot.html'
    offline.plot(fig, filename=name)


def mean_std_visualisation(filename):
    with open(filename, 'r') as jsonfile:
        mean_std = json.load(jsonfile)
    data = []
    for item in mean_std.keys():
        if item != 'Holiday':
            trace = go.Scatter(x=list(range(0, 24)), y=[mean_std[item][i][0] for i in range(24)],
                               mode='markers', marker=dict(size=4 * [mean_std[item][i][0] for i in range(24)]),
                               name='Weekday-' + item)
        else:
            trace = go.Scatter(x=list(range(0, 24)), y=[mean_std[item][i][0] for i in range(24)],
                               mode='markers', marker=dict(size=4 * [mean_std[item][i][0] for i in range(24)]),
                               name=item)
        data.append(trace)
    layout = go.Layout(title='price difference mean std visualisation', xaxis={"title": "Hour"},
                       yaxis={"title": "Mean price difference [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'price_difference_mean_std.html'
    offline.plot(fig, filename=name)


def errorvisualisation(inputfilepath1, inputfilepath2, inputfilepath3, inputfilepath4, inputfilepath5):
    df1 = pd.read_csv(inputfilepath1, sep=';', index_col=['timestamp'])  # sarimax
    df2 = pd.read_csv(inputfilepath2, sep=';', index_col=['timestamp'])  # dnn-MAE
    df3 = pd.read_csv(inputfilepath3, sep=';', index_col=['timestamp'])  # sarimax-2
    df4 = pd.read_csv(inputfilepath4, sep=';', index_col=['timestamp'])  # dnn-MAE + sarimax
    inputdf = pd.read_csv(inputfilepath5, sep=';', index_col=['UTC Timestamp'])
    year = df1.index.to_list()[0].split(' ')[0].split('-')[0]
    df1['price estimator Model'] = inputdf.loc[(year + '-01-01 00'):(year + '12-31 23'),
                                   'price estimator Model'].values
    df1['price electricity real'] = inputdf.loc[(year + '-01-01 00'):(year + '12-31 23'),
                                    'price electricity real'].values
    df1['price after correction'] = df1['prediction'] + df1['price estimator Model']
    df2['price after correction'] = df2['prediction'] + df1['price estimator Model']
    df3['price after correction'] = df3['prediction'] + df1['price estimator Model']
    df4['price after correction'] = df4['prediction'] + df1['price estimator Model']
    data = []
    timestamp = df1.index.to_list()
    error = df4['error']
    mae_error = np.around(np.mean(np.abs(error)), decimals=3)
    actual = df1['actual']
    mae_actual = np.around(np.mean(np.abs(actual)), decimals=3)
    fundamental = df1['price estimator Model']
    aftercorrection1 = df1['price after correction']
    aftercorrection2 = df2['price after correction']
    aftercorrection3 = df3['price after correction']
    aftercorrection4 = df4['price after correction']
    real = df1['price electricity real']
    prediction1 = df1['prediction']
    prediction2 = df2['prediction']
    prediction3 = df3['prediction']
    prediction4 = df4['prediction']
    maesarimax = np.round(np.mean(np.absolute(df1.error)), decimals=3)
    maednn1 = np.round(np.mean(np.absolute(df2.error)), decimals=3)
    maednn2 = np.round(np.mean(np.absolute(df3.error)), decimals=3)
    maesarimaxdnn = np.round(np.mean(np.absolute(df4.error)), decimals=3)
    trace1 = go.Scatter(x=timestamp, y=actual, name='Price difference fundamental Model')
    trace2 = go.Scatter(x=timestamp, y=prediction1, name='Price difference Prediction SARIMAX')
    trace3 = go.Scatter(x=timestamp, y=prediction2, name='Price difference Prediction DNN-MAE')
    trace4 = go.Scatter(x=timestamp, y=prediction3, name='Price difference Prediction SARIMAX AIC')
    trace5 = go.Scatter(x=timestamp, y=prediction4, name='Price difference Prediction DNN + SARIMAX')
    trace6 = go.Scatter(x=timestamp, y=fundamental, name='Price fundamental model (MAE: ' + str(mae_actual) + ')')
    trace7 = go.Scatter(x=timestamp, y=aftercorrection1,
                        name='Price after correction SARIMAX (MAE: ' + str(maesarimax) + ')')
    trace8 = go.Scatter(x=timestamp, y=aftercorrection2,
                        name='Price after correction DNN-MAE (MAE: ' + str(maednn1) + ')')
    trace9 = go.Scatter(x=timestamp, y=aftercorrection3,
                        name='Price after correction SARIMAX AIC (MAE: ' + str(maednn2) + ')')
    trace10 = go.Scatter(x=timestamp, y=aftercorrection4,
                         name='Price after correction DNN + SARIMAX  (MAE: ' + str(maesarimaxdnn) + ')')
    trace11 = go.Scatter(x=timestamp, y=real, name='Price real')
    # data.append(trace1)
    # data.append(trace2)
    # data.append(trace3)
    # data.append(trace4)
    # data.append(trace5)
    data.append(trace6)
    data.append(trace7)
    data.append(trace8)
    data.append(trace9)
    data.append(trace10)
    data.append(trace11)
    layout = go.Layout(
        title='Price after correction visualisation ' + year,
        xaxis={"title": "Time"},
        yaxis={"title": "Mean Absolute Error [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'correction_visualisation_comparison_SARIMAX_AIC' + year + '.html'
    offline.plot(fig, filename=name)


def predictionvisualisation(inputfilepath):
    df = pd.read_csv(inputfilepath, sep=';', index_col=['timestamp'])
    year = df.index.to_list()[0].split(' ')[0].split('-')[0]
    data = []
    timestamp = df.index.to_list()
    prediction = df['prediction']
    mae_error = np.around(np.mean(np.abs(prediction)), decimals=3)
    actual = df['actual']
    mae_actual = np.around(np.mean(np.abs(actual)), decimals=3)
    trace1 = go.Scatter(x=timestamp, y=prediction, name='Prediction')
    trace2 = go.Scatter(x=timestamp, y=actual, name='Actual')
    data.append(trace1)
    data.append(trace2)
    layout = go.Layout(title='Prediction visualisation ' + year, xaxis={"title": "Time"},
                       yaxis={"title": "Mean Absolute Error [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'prediction_visualisation' + year + '.html'
    offline.plot(fig, filename=name)


def errorcompare(inputfilepath1, inputfilepath2):
    df1 = pd.read_csv(inputfilepath1, sep=';', index_col=['timestamp'])
    df2 = pd.read_csv(inputfilepath2, sep=';', index_col=['timestamp'])
    year = df1.index.to_list()[0].split(' ')[0].split('-')[0]
    data = []
    timestamp = df1.index.to_list()
    error1 = abs(df1['error'])
    error2 = abs(df2['error'])
    mae_error1 = np.around(np.mean(np.abs(error1)), decimals=3)
    mae_error2 = np.around(np.mean(np.abs(error2)), decimals=3)
    actual = df1['actual']
    mae_actual = np.around(np.mean(np.abs(actual)), decimals=3)
    prediction1 = df1['prediction']
    prediction2 = df2['prediction']
    trace1 = go.Scatter(x=timestamp, y=error1, name='Error1 after correction: ' + str(mae_error1))
    trace2 = go.Scatter(x=timestamp, y=actual, name='Actual: ' + str(mae_actual))
    trace3 = go.Scatter(x=timestamp, y=prediction1, name='Prediction1')
    trace4 = go.Scatter(x=timestamp, y=error2, name='Error2 after correction: ' + str(mae_error2))
    trace5 = go.Scatter(x=timestamp, y=prediction2, name='Prediction2')
    data.append(trace1)
    data.append(trace2)
    data.append(trace3)
    data.append(trace4)
    data.append(trace5)
    layout = go.Layout(title='Error visualisation ' + year, xaxis={"title": "Time"},
                       yaxis={"title": "Mean Absolute Error [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'error_compare' + year + '.html'
    offline.plot(fig, filename=name)


def maecomparison(inputfilepath1, inputfilepath2):
    df1 = pd.read_csv(inputfilepath1, sep=';')
    df2 = pd.read_csv(inputfilepath2, sep=';')
    days = int(df1.shape[0] / 24)
    timestamplist = []
    mae_v1 = []
    mae_v2 = []
    mae_actual = []
    for i in range(days):
        timestamplist.append(df1.loc[i * 24, 'timestamp'].split(' ')[0])
        mae1 = np.mean(np.absolute(df1.loc[i * 24: i * 24 + 23, 'error']))
        actual = np.mean(np.absolute(df1.loc[i * 24: i * 24 + 23, 'actual']))
        mae_v1.append(mae1)
        mae2 = np.mean(np.absolute(df2.loc[i * 24: i * 24 + 23, 'error']))
        mae_v2.append(mae2)
        mae_actual.append(actual)
    timestamplist = [datetime.strptime(item, '%Y-%m-%d').date() for item in timestamplist]
    trace1 = go.Scatter(x=timestamplist, y=mae_v1, name='MAE Ficxed')
    trace2 = go.Scatter(x=timestamplist, y=mae_v2, name='MAE Model t_3')
    trace3 = go.Scatter(x=timestamplist, y=mae_actual, name='MAE actual')
    data = []
    data.append(trace1)
    data.append(trace2)
    data.append(trace3)
    layout = go.Layout(title='Error visualisation ', xaxis={"title": "Time"},
                       yaxis={"title": "Mean Absolute Error [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'mae_compare' + '.html'
    offline.plot(fig, filename=name)



def pricecomparison(inputfile1, inputfile2):
    df1 = pd.read_csv(inputfile1, sep=';', index_col=['UTC Timestamp'])
    df2 = pd.read_csv(inputfile2, sep=';', index_col=['timestamp'])
    df1 = df1.loc['2018-01-01 00':, :]
    timestamp = df1.index.to_list()
    priceaverage = df1.loc[:, 'Price Average']
    priceprediction = df2.loc[:, 'prediction']
    real = df2.loc[:, 'actual']
    data = []
    trace1 = go.Scatter(x=timestamp, y=priceaverage, name='Price Average')
    trace2 = go.Scatter(x=timestamp, y=priceprediction, name='Price Prediction')
    trace3 = go.Scatter(x=timestamp, y=real, name='Price real')
    data.append(trace1)
    data.append(trace2)
    data.append(trace3)
    layout = go.Layout(
        title='Price  visualisation ' + '2018',
        xaxis={"title": "Time"},
        yaxis={"title": "Price [€/MWh]"})
    fig = go.Figure(data=data, layout=layout)
    name = 'Visualisation/' + 'price_vis' + '2018' + '.html'
    offline.plot(fig, filename=name)

