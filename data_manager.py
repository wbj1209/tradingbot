import os
import pandas as pd
import numpy as np

import settings

# preprocess 데이터 전처리
# load_data 데이터 불러오기

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',

    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio'
]


def preprocess(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data['close_ma{}'.format(window)] = data['close'].rolling(
            window).mean()
        data['volume_ma{}'.format(window)] = data['volume'].rolling(
            window).mean()
        data['close_ma%d_ratio' % window] = (
            data['close'] - data['close_ma%d' % window]) / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = (
            data['volume'] - data['volume_ma%d' % window]) / data['volume_ma%d' % window]

    # 문의
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = (
        data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values

    data['high_close_ratio'] = (
        data['high'].values - data['close'].values) / data['close'].values

    data['low_close_ratio'] = (
        data['low'].values - data['close'].values) / data['close'].values

    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = (
        data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values

    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = ((data['volume'][1:].values - data['volume'][:-1].values) /
                                               data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values)

    return data


def load_data(date_from, date_to):
    data = pd.read_csv(
        os.path.join(settings.BASE_DIR,
                     'data/{}/{}.csv'.format(args.ver, stock_code)),
        thousands=',', header=header, converters={'date': lambda x: str(x)})

    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    data = data.sort_values(by='date').reset_index()

    # 데이터 전처리
    data = preprocess(data)

    # 기간 필터링
    data['date'] = data['date'].str.replace('-', '')
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    data = data.dropna()

    # 차트 데이터, 학습데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]
    training_data = data[COLUMNS_TRAINING_DATA_V1]

    return chart_data, training_data
