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
    time_end = dt.datetime(Year, Month, Day, 15, 0, 0)
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


f = lambda x: (x['AskPrice1'] + x['BidPrice1']) / 2 if x['AskPrice1'] > 0 and x['BidPrice1'] > 0 else np.nan    


def bar_generator(data, freq='3S'):

    if len(data) == 0:
        return None
    else:
        function_dict = {'PreClose' :'first', 'TodayOpen' :'first', 'Close' :'last', 'Open' :'first', 'High' :'max', 
                         'Low' :'min', 'Status' :'last', 'AskPrice1' :'last', 'AskPrice2' :'last', 'AskPrice3' :'last', 
                         'AskPrice4' :'last', 'AskPrice5' :'last', 'AskPrice6' :'last', 'AskPrice7' :'last', 
                         'AskPrice8' :'last', 'AskPrice9' :'last', 'AskPrice10' :'last', 'BidPrice1' :'last', 
                         'BidPrice2' :'last', 'BidPrice3' :'last', 'BidPrice4' :'last', 'BidPrice5' :'last', 
                         'BidPrice6' :'last', 'BidPrice7' :'last', 'BidPrice8' :'last', 'BidPrice9' :'last', 
                         'BidPrice10' :'last', 'AskVol1' :'last', 'AskVol2' :'last', 'AskVol3' :'last', 
                         'AskVol4' :'last', 'AskVol5' :'last', 'AskVol6' :'last', 'AskVol7' :'last', 'AskVol8' :'last', 
                         'AskVol9' :'last', 'AskVol10' :'last', 'BidVol1' :'last', 'BidVol2' :'last', 'BidVol3' :'last', 
                         'BidVol4' :'last', 'BidVol5' :'last', 'BidVol6' :'last', 'BidVol7' :'last', 'BidVol8' :'last', 
                         'BidVol9' :'last', 'BidVol10' :'last', 'WeightedAvgBidPrice' :'last', 'WeightedAvgAskPrice' :'last', 
                         'TransactionNum' :'sum', 'TransactionVol' :'sum', 'TransactionAmount' :'sum', 'TotalAskVol' :'sum', 
                         'TotalBidVol' :'sum', 'HighLimit' :'last', 'LowLimit' :'last'}
        result = data.resample(freq, closed='right', label='right').agg(function_dict)
        result['Mid'] = result.apply(f, axis=1)
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


def TargerGeneratorReturn(data, future_period):

    base_freq_str = data.name
    freq = int(base_freq_str[:-1])
    num_shift = int(future_period / freq)
    result = pd.DataFrame()
    Mid_shift = data['Mid'].shift(-num_shift)
    MidReturn = (Mid_shift - data['Mid']) / (data['Mid'])
    MidReturn[~np.isfinite(MidReturn)] = np.nan
    result['MidReturn' + str(future_period)] = MidReturn
    return result


def TargerGeneratorReturn_New(data, future_period):

    base_freq_str = '3S'
    freq = int(base_freq_str[:-1])
    num_shift = int(future_period / freq)
    result = pd.DataFrame()
    Mid_shift = data['Mid'].shift(-num_shift)
    MidReturn = (Mid_shift - data['Mid']) / (data['Mid'])
    MidReturn[~np.isfinite(MidReturn)] = np.nan
    result['MidReturn' + str(future_period)] = MidReturn
    return result


def TargetSeriesGenerate(data, rt_series=[15, 60, 120, 180, 300]):

    result = []
    for ti in rt_series:
        result.append(TargerGeneratorReturn(data, ti))
    target_all = pd.concat(result, axis=1)
    return target_all


def TargetSeriesGenerate_NEW(data, rt_series=[15, 60, 120, 180, 300]):

    result = []
    result_diff = []
    result_diff_name = []
    for ix, ti in enumerate(rt_series):
        result.append(TargerGeneratorReturn_New(data, ti))
        if ix > 0:
            new_diff = pd.Series(result[-1].iloc[:, 0].values - result[-2].iloc[:, 0].values, index=result[-1].index)
            result_diff.append(new_diff)
            result_diff_name.append('Diff_Return_{}-{}'.format(rt_series[ix], rt_series[ix - 1]))
    target_time_all = pd.concat(result, axis=1)
    target_diff_all = pd.concat(result_diff, axis=1)
    target_diff_all.columns = result_diff_name
    target_all = pd.concat((target_time_all, target_diff_all), axis=1)
    return target_all


def bar_h5_process(multi_proc_ticker, h5_path, h5_name, tick_save_path, target_save_path, freq='3S'):
    print(h5_name)
    trading_day = int(h5_name[:-3])
    tick_save_name = tick_save_path + '/' + h5_name
    target_save_name = target_save_path + '/' + h5_name
    h5_file_path = h5_path + '/' + h5_name
    h5s = pd.HDFStore(h5_file_path, 'r')
    keys = h5s.keys()
    if multi_proc_ticker:
        p = Pool(multi_proc_ticker)
        for ticker in keys:
            p.apply_async(multi_proc_tickers_tick_only, args=(h5s[ticker], ticker, trading_day, tick_save_name, target_save_name, freq,))
        p.close()
        p.join()
    else:
        for ticker in keys:
            multi_proc_tickers_tick_only(h5s[ticker], ticker, trading_day, tick_save_name, target_save_name, freq)


def bar_h5_process_with_existing_tick(multi_proc_ticker, tick_bar, h5_name, tick_save_path, target_save_path, freq='3S'):
    print(h5_name)
    trading_day = int(h5_name[:-3])
    target_save_name = target_save_path + '/' + h5_name
    h5_file_path = tick_bar + '/' + h5_name
    h5s = pd.HDFStore(h5_file_path, 'r')
    keys = h5s.keys()
    if multi_proc_ticker:
        p = Pool(multi_proc_ticker)
        for ticker in keys:
            p.apply_async(multi_proc_ticker_with_existing_bars, args=(ticker, trading_day, target_save_name, freq,))
        p.close()
        p.join()
    else:
        for ticker in keys:
            multi_proc_ticker_with_existing_bars(ticker, trading_day, target_save_name, freq)
    h5s.close()


def multi_proc_tickers(data, ticker, trading_day, tick_save_name, target_save_name, freq):
    print(ticker)
    # warnings.filterwarnings('ignore')
    data_cleaned = data_clean(data, trading_day)
    bars = bar_generator(data_cleaned, freq)
    if bars is not None:
        targets = TargetSeriesGenerate(bars)
        bars.to_hdf(tick_save_name, key=ticker, mode='a')
        targets.to_hdf(target_save_name, key=ticker, mode='a')
    else:
        pass


def multi_proc_tickers_tick_only(data, ticker, trading_day, tick_save_name, target_save_name, freq):
    print(ticker)
    # warnings.filterwarnings('ignore')
    data_cleaned = data_clean(data, trading_day)
    bars = bar_generator(data_cleaned, freq)
    if bars is not None:
        bars.to_hdf(tick_save_name, key=ticker, mode='a')
    else:
        pass


def multi_proc_ticker_with_existing_bars(ticker, trading_day, target_save_name, freq):
    print(ticker)

    bars = pd.read_hdf('F:/tick_bar/{}.h5'.format(trading_day), key=ticker)
    if bars is not None:
        targets = TargetSeriesGenerate_NEW(bars)
        targets.to_hdf(target_save_name, key=ticker, mode='a')
    else:
        pass



def need_dates(data_file_path, begin_date, end_date):

    h5_list = np.array(os.listdir(data_file_path))
    h5_list.sort()
    need_days = h5_list[(h5_list >= '{}.h5'.format(begin_date)) & (h5_list <= '{}.h5'.format(end_date))]
    return need_days


class tick_bar_generator():

    def __init__(self, multi_proc, multi_proc_ticker, h5_path, tick_save_path, target_save_path, begin_date, end_date, freq):

        self.multi_proc = multi_proc
        self.multi_proc_ticker = multi_proc_ticker
        self.h5_path = h5_path
        self.tick_save_path = tick_save_path
        self.target_save_path = target_save_path
        self.begin_date = begin_date
        self.end_date = end_date
        self.freq = freq
        self.tick_bar = tick_save_path


    def multi_tick_bar_generator(self):

        multi_proc = self.multi_proc
        multi_proc_ticker = self.multi_proc_ticker
        h5_path = self.h5_path
        tick_save_path = self.tick_save_path
        target_save_path = self.target_save_path
        begin_date = self.begin_date
        end_date = self.end_date
        freq = self.freq

        if not os.path.exists(tick_save_path):
            os.makedirs(tick_save_path)
        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path)
        needed_dates = need_dates(h5_path, begin_date, end_date)
        print(needed_dates)
        if multi_proc:
            p = Pool(multi_proc)
            for day in needed_dates:
                p.apply_async(bar_h5_process, args=(multi_proc_ticker, h5_path, day, tick_save_path, target_save_path, freq,))
            p.close()
            p.join()
        else:
            for day in needed_dates:
                bar_h5_process(multi_proc_ticker, h5_path, day, tick_save_path, target_save_path, freq)


    def multi_bar_target_generator_with_existing_data(self):

        multi_proc = self.multi_proc
        multi_proc_ticker = self.multi_proc_ticker
        tick_bar = self.tick_bar
        tick_save_path = self.tick_save_path
        target_save_path = self.target_save_path
        begin_date = self.begin_date
        end_date = self.end_date
        freq = self.freq


        if not os.path.exists(tick_save_path):
            os.makedirs(tick_save_path)
        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path)
        needed_dates = need_dates(h5_path, begin_date, end_date)
        print(needed_dates)
        if multi_proc:
            p = Pool(multi_proc)
            for day in needed_dates:
                p.apply_async(bar_h5_process_with_existing_tick, args=(multi_proc_ticker, tick_bar, day, tick_save_path, target_save_path, freq,))
            p.close()
            p.join()
        else:
            for day in needed_dates:
                bar_h5_process_with_existing_tick(multi_proc_ticker, tick_bar, day, tick_save_path, target_save_path, freq)


 
if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')
    h5_path = 'F:/tick_raw'
    tick_bar = 'F:/tick_bar'
    targets_bar = 'F:/targets_bar_new_add'
    bgd = "20180830"
    edd = "20180831"
    freq = "3S"
    multi_proc = 0
    multi_proc_ticker = 0
    tick_bar_target_ob = tick_bar_generator(multi_proc, multi_proc_ticker, h5_path, tick_bar, targets_bar, bgd, edd, freq)
    # tick_bar_target_ob.multi_tick_bar_generator()
    tick_bar_target_ob.multi_bar_target_generator_with_existing_data()

