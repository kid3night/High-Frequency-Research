import pandas as pd
import numpy as np
import datetime as dt
import os
from multiprocessing import Pool
import warnings


def slice_date(date):
    day = date % 100
    month = date // 100 % 100
    year = date // 10000
    return (year, month, day)


def slice_min(Time):
    if np.isnan(Time):
        return(-1, -1, -1)
    else:
        Hour = Time // 10000000
        Min = Time // 100000 % 100
        Sec = Time // 1000 % 100
        return (int(Hour), int(Min), int(Sec))


def data_clean(data, trading_day):

    # clean data into same time range
    (Year, Month, Day) = slice_date(trading_day)
    time_start = dt.datetime(Year, Month, Day, 9, 15, 0)
    time_end = dt.datetime(Year, Month, Day, 15, 01, 0)
    time_range = pd.date_range(time_start, time_end, freq='1S')
    time_range2 = pd.date_range(time_start, time_end, freq='3S')
    time_col = data['Time'].values
    temp_dateTime = []
    for i in range(len(time_col)):
        H, M, S = slice_min(time_col[i])
        if H == -1:
            continue
        else:
            temp_dateTime.append(dt.datetime(Year, Month, Day, H, M, S))
    data.index = pd.DatetimeIndex(temp_dateTime)

    main_part_cols = ['High', 'Low', 'Close',
                      'TransactionNum', 'TransactionVol', 'TransactionAmount', 'TotalBidVol',
                      'TotalAskVol', 'WeightedAvgBidPrice', 'WeightedAvgAskPrice',
                      'HighLimit', 'LowLimit', 'AskPrice1', 'AskPrice2', 'AskPrice3',
                      'AskPrice4', 'AskPrice5', 'AskPrice6', 'AskPrice7', 'AskPrice8',
                      'AskPrice9', 'AskPrice10', 'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4',
                      'AskVol5', 'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10',
                      'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
                      'BidPrice6', 'BidPrice7', 'BidPrice8', 'BidPrice9', 'BidPrice10',
                      'BidVol1', 'BidVol2', 'BidVol3', 'BidVol4', 'BidVol5', 'BidVol6',
                      'BidVol7', 'BidVol8', 'BidVol9', 'BidVol10']
    sub_cols = ['Status', 'PreClose', 'Open']

    # diminish the same data points at the same timestamps
    # When encountered mutiple row for the same time_index,
    # take mean on main part, take last on PreClose, TodayOpen, and Status
    data_main_part = data[main_part_cols].groupby(level=0).mean().astype(int)
    data_status = data[sub_cols].groupby(level=0).last()
    data_status.columns = ['Status', 'PreClose', 'TodayOpen']

    result = pd.DataFrame()
    result[data_main_part.columns] = data_main_part
    result[data_status.columns] = data_status 
    # result = result.reindex(time_range)
    # To avoid lossing info because of differnt sampling time point
    result = result.reindex(time_range).ffill().dropna()
    result = result.reindex(time_range2).ffill().dropna().astype(int)
    # create open col using close of last tick
    result['Open'] = result['Close'].shift(1).fillna(0).astype(int)
    # take difference on these three cumulated volume to get vols in last tick
    result[['TransactionNum', 'TransactionVol','TransactionAmount']] = result[['TransactionNum', 'TransactionVol','TransactionAmount']].diff(1).dropna()
    result = result.dropna().astype(int)
    UpThreshold = dt.datetime(Year, Month, Day, 9, 30, 0)
    DownThreshold = dt.datetime(Year, Month, Day, 11, 30, 0)
    UpThreshold2 = dt.datetime(Year, Month, Day, 13, 0, 0)
    result = result[((result.index > UpThreshold) & (result.index <= DownThreshold)) | (result.index > UpThreshold2)]
    if len(result) > 0:
        result['Open'].iloc[0] = result['TodayOpen'].iloc[0]
    result.name = UpThreshold
    return result





def bar_generator(data, freq='3S'):

    if len(data) == 0:
        return None
    else:

        function_dict = {'Open' :'first', 'TransactionNum' :'sum', 'TransactionVol' :'sum', 'TransactionAmount' :'sum'}
        result_part1 = data.resample(freq, closed='left', label='right').agg(function_dict)
        result_part2 = data['Close'].resample(freq, closed='left', label='right').ohlc()
        result
        result = result.ffill().bfill()
        Year, Month, Day = data.index[0].year, data.index[0].month, data.index[0].day
        UpThreshold = dt.datetime(Year, Month, Day, 9, 30, 0)
        DownThreshold = dt.datetime(Year, Month, Day, 11, 30, 0)
        UpThreshold2 = dt.datetime(Year, Month, Day, 13, 0, 0)
        stk_columns = ['Status', 'PreClose', 'TodayOpen', 'Open', 'High', 'Low', 'Close', 'Mid', 'TransactionNum',
                       'TransactionVol', 'TransactionAmount', 'TotalBidVol', 'TotalAskVol', 'WeightedAvgBidPrice', 
                       'WeightedAvgAskPrice', 'HighLimit', 'LowLimit', 'AskPrice1', 'AskPrice2', 'AskPrice3', 
                       'AskPrice4', 'AskPrice5', 'AskPrice6', 'AskPrice7', 'AskPrice8', 'AskPrice9', 'AskPrice10', 
                       'AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5', 'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 
                       'AskVol10', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 'BidPrice6', 
                       'BidPrice7', 'BidPrice8', 'BidPrice9', 'BidPrice10', 'BidVol1', 'BidVol2', 'BidVol3', 
                       'BidVol4', 'BidVol5', 'BidVol6', 'BidVol7', 'BidVol8', 'BidVol9', 'BidVol10']
        result = result[((result.index > UpThreshold) & (result.index <= DownThreshold)) | (result.index > UpThreshold2)]
        result = result.loc[:, stk_columns]
        result.name = freq
        return result