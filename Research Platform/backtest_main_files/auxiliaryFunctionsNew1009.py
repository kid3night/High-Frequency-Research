import pandas as pd
import numpy as np
import scipy.stats as sts
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import numba
from numba import jit
import gc


def featurePreprocess(_stock, _target):

    finite_pos = np.isfinite(_target)

    # get non-nan, inf data position
    pos1 = _stock['Status'] == 'O'
    pos2 = np.logical_and.reduce(finite_pos, axis=1)
    posNotnan = pos1 & pos2

    return _stock.iloc[posNotnan.values], _target.iloc[posNotnan.values]


def targetSelect(_target):

    return _target[['MidReturn15', 'MidReturn30', 'MidReturn60'
                    'MidReturn90', 'MidReturn300', 'MidReturn600',
                    'MidReturn1500', 'MidReturn2400']]



def fetchData(tick_path, target_path, order_path, transaction_path, orderbook_path,
              ticker, startTime, endTime, targetScope, feature_type, each_type_data_need):


    data_feed = dict()
    data_affiliated = dict()

    target = pd.HDFStore(target_path, 'r')
    index_select = "index>pd.Timestamp(\'{}\') & index<pd.Timestamp(\'{}\')".format(startTime, endTime)
    target_data_raw = target.select(ticker, index_select)
    tick = pd.HDFStore(tick_path, 'r')
    status_col = "columns ={}".format(['Status', 'AskPrice1', 'BidPrice1', 'Mid', 'HighLimit', 'LowLimit', 'Close'])
    status_data = tick.select(ticker, index_select + "&" + status_col)

    if 'tick' in feature_type:
        col_select_tick = "columns ={}".format(each_type_data_need['tick'])
        tick_data_raw = tick.select(ticker, index_select + "&" + col_select_tick)

    if 'order' in feature_type:
        order = pd.HDFStore(order_path, 'r')
        col_select_order = "columns ={}".format(each_type_data_need['order'])
        order_data_raw = order.select(ticker, index_select + "&" + col_select_order)
        order.close()

    if 'transaction' in feature_type:
        transaction = pd.HDFStore(transaction_path, 'r')
        col_select_transaction = "columns ={}".format(each_type_data_need['transaction'])
        transaction_data_raw = transaction.select(ticker, index_select + "&" + col_select_transaction)
        transaction.close()

    if 'orderbook' in feature_type:
        orderbook = pd.HDFStore(orderbook_path, 'r')
        col_select_orderbook = "columns ={}".format(each_type_data_need['orderbook'])
        orderbook_data_raw = orderbook.select(ticker, index_select + "&" + col_select_orderbook)
        orderbook.close()

    time_range = status_data.index.hour * 100  + status_data.index.minute
    pos1 = status_data['Status'] == 79
    pos2 = (time_range > 935) & (time_range < 1455)
    intersect_pos = pos1 & pos2

    data_affiliated['level1_data'] = status_data.loc[intersect_pos, ['AskPrice1', 'BidPrice1', 'Mid', 'HighLimit', 'LowLimit', 'Close']]
    data_affiliated['target_data'] = target_data_raw.loc[intersect_pos, :]
    data_affiliated['status'] = status_data.loc[intersect_pos, ['Status']]

    if 'order' in feature_type or 'orderbook' in feature_type:
        if 'order' in feature_type:
            pos_for_orders = intersect_pos.reindex(order_data_raw.index).ffill().fillna(False)
        else:
            pos_for_orders = intersect_pos.reindex(orderbook_data_raw.index).ffill().fillna(False)
    if 'transaction' in feature_type:
        pos_for_transactions = intersect_pos.reindex(transaction_data_raw.index).ffill().fillna(False)

    if 'tick' in feature_type:
        data_feed['tick_data'] = tick_data_raw.loc[intersect_pos, :]
    if 'order' in feature_type:
        data_feed['order_data'] = order_data_raw.loc[pos_for_orders, :]
    if 'transaction' in feature_type:
        data_feed['transaction_data'] = transaction_data_raw.loc[pos_for_transactions, :]
    if 'orderbook' in feature_type:
        data_feed['orderbook_data'] = orderbook_data_raw.loc[pos_for_orders, :]

    tick.close()
    target.close()

    gc.collect()
    del gc.garbage[:]
    gc.collect()

    return data_feed, data_affiliated



def fetchData_incomplete(tick_path, target_path, ticker, startTime, endTime, targetScope):

    data_feed = dict()
    tick = pd.HDFStore(tick_path, 'r')
    target = pd.HDFStore(target_path, 'r')
    index_select = 'index>pd.Timestamp(\'{}\') & index<pd.Timestamp(\'{}\')'.format(startTime, endTime + 1)
    tick_data_raw = tick.select(ticker, index_select)
    target_data_raw = target.select(ticker, index_select)
    time_range = (tick_data_raw.index.hour * 100  + tick_data_raw.index.minute).values
    pos1 = (tick_data_raw['Status'] == 79).values
    pos2 = (time_range > 930) & (time_range < 1455)
    intercept = pos1 & pos2
    data_feed['tick_data'] = tick_data_raw.loc[intercept, :]
    target_data = target_data_raw.loc[intercept, targetScope]
    tick.close()
    target.close()
    return data_feed, target_data


def feature_intersect(feature, target, stock):

    _pos_f = np.isfinite(feature)
    _pos_target


def featureAftprocess(feature, _target, _stock):

    finite_pos = np.isfinite(feature)

    return feature.iloc[finite_pos.values], _target.iloc[finite_pos.values], _stock.iloc[finite_pos.values]


def percentile_stats(signal, ret, start=10, stride=10):

    percentiles = np.percentile(signal, range(start, 100, stride))
    percs = np.searchsorted(percentiles, signal, side='left')
    df = pd.DataFrame({'rets': ret, 'sigs': percs})
    return_percent = df.groupby('sigs').mean()
    return return_percent


def curvePlotSeries(feature, _target, feature_name, target_name):

    ixsort = feature.values.argsort()
    fig, axs = plt.subplots(4, 3, figsize=(12, 9), dpi=120)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    for i, ix in enumerate(_target):
        retSorted = (_target[ix].values - _target[ix].values.mean())[ixsort]
        corr = np.corrcoef(feature.values, _target[ix].values)
        spearmanr = sts.spearmanr(feature.values, _target[ix].values)
        cor = round(corr[0, 1], 3)
        scor = round(spearmanr[0], 3)
        axs[i // 3, i % 3].plot(np.arange(len(feature)), retSorted.cumsum())
        axs[i // 3, i % 3].set_title(ix)
    plt.savefig(feature_name + target_name + '.png')
    plt.clf()


def percentilePlotSeries(feature, _target, feature_name, target_name, start=5, stride=5):
    fig, axs = plt.subplots(4, 3, figsize=(12, 9), dpi=120)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    for i, ix in enumerate(_target):
        return_percent = percentile_stats(feature, _target[ix], start, stride)
        axs[i // 3, i % 3].bar(return_percent.index, return_percent.values.T[0])
        axs[i // 3, i % 3].set_title(ix)
    plt.savefig(feature_name + target_name + '.png')
    plt.clf()

def curvePlotSingle(feature, target, corr_pearson, corr_spearman):

    ixsort = feature.values.argsort()
    ret_sorted = (target.values - target.values.mean())[ixsort]
    corr = np.corrcoef(feature.values, target.values)
    spearmanr = sts.spearmanr(feature.values, target.values)
    cor = round(corr_pearson, 3)
    scor = round(corr_spearman, 3)
    return ret_sorted


def return_sort_single(feature, target):

    ixsort = feature.values.argsort()
    ret_sorted = (target.values - target.values.mean())[ixsort]
    return ret_sorted


def percentilePlotSingle(feature, target, start=5, stride=5):

    return_percent = percentile_stats(feature, target, start, stride)

    return return_percent    

def transform(signal, lower_percentile, upper_percentile):
    """
    @signal: should either be a pandas Series or a numpy array
    """
    try:
        lower, upper = np.percentile(np.abs(signal.values), [abs(lower_percentile), abs(upper_percentile)])
    except:

        lower, upper = np.percentile(np.abs(signal), [abs(lower_percentile), abs(upper_percentile)])
    if lower_percentile < 0:
        lower = -lower
    if upper_percentile < 0:
        upper = -upper
    return lower, upper


# @numba.jit(nopython=True)
# def signal2positionNumba(signal, upper, lower, initial_pos=0.0):
#     """
#     signal must be numpy 1-d array
#     upper>lower>0 are two constants

#     @adjust_position_size: delta / atr_adjust,
#         an array of already adjusted contract position size estimation
#     """

#     pos = np.zeros(signal.size, dtype=np.float64)
#     pos[-1] = initial_pos
#     for i in range(signal.size):
#         if pos[i - 1] > 1e-10:
#             if signal[i] <= lower:
#                 pos[i] = 0.
#             else:
#                 pos[i] = pos[i - 1]
#         elif pos[i - 1] < -1e-10:
#             if signal[i] >= -lower:
#                 pos[i] = 0.
#             else:
#                 pos[i] = pos[i - 1]
#         else:
#             if signal[i] >= upper:
#                 pos[i] = 1.0
#             elif signal[i] <= -upper:
#                 pos[i] = -1.0

#     return pos


@numba.jit(nopython=True)
def signal2positionNumba(signal, upper, lower, initial_pos=0.0):
    """
    signal must be numpy 1-d array
    upper>lower>0 are two constants

    @adjust_position_size: delta / atr_adjust,
        an array of already adjusted contract position size estimation
    """

    pos = np.zeros(signal.size, dtype=np.float64)
    pos[-1] = initial_pos
    for i in range(signal.size):
        if pos[i - 1] > 1e-10:
            if signal[i] <= -lower:
                pos[i] = 0.
            else:
                pos[i] = pos[i - 1]
        elif pos[i - 1] < -1e-10:
            if signal[i] >= lower:
                pos[i] = 0.
            else:
                pos[i] = pos[i - 1]
        else:
            if signal[i] >= upper:
                pos[i] = 1.0
            elif signal[i] <= -upper:
                pos[i] = -1.0

    return pos



def signal2positionNumba_groupby(signal, lower_percentile, upper_percentile, initial_pos=0.0):
    """
    @signal: should either be a pandas Series or a numpy array
    """
    if len(signal) < 1:
        return np.array([])
    try:
        lower, upper = np.percentile(np.abs(signal.values), [abs(lower_percentile), abs(upper_percentile)])
    except:
        lower, upper = np.percentile(np.abs(signal), [abs(lower_percentile), abs(upper_percentile)])
    if lower_percentile < 0:
        lower = -lower
    if upper_percentile < 0:
        upper = -upper

    res = signal2positionNumba(signal.values, upper, lower)
    return res


def signal2positionNumba_groupby_nonzero(signal, lower_percentile, upper_percentile, initial_pos=0.0):
    """
    @signal: should either be a pandas Series or a numpy array
    """
    if len(signal) < 1:
        return np.array([])
    temp_signal = signal.loc[signal != 0]
    if len(temp_signal) < 1:
        lower, upper = 0, 0
    else:
        try:
            lower, upper = np.percentile(np.abs(temp_signal.values), [abs(lower_percentile), abs(upper_percentile)])
        except:
            lower, upper = np.percentile(np.abs(temp_signal), [abs(lower_percentile), abs(upper_percentile)])
        if lower_percentile < 0:
            lower = -lower
        if upper_percentile < 0:
            upper = -upper

    # res = signal2positionNumba(signal.values, lower, upper)
    res = signal2positionNumba(signal.values, 0.0015, 0.0005)
    # res = signal2positionNumba(signal.values, upper, lower)
    return res


# @numba.jit(nopython=True)
# def signal2positionNumba(signal, upper, lower, initial_pos=0.0):
#     """
#     signal must be numpy 1-d array
#     upper>lower>0 are two constants

#     @adjust_position_size: delta / atr_adjust,
#         an array of already adjusted contract position size estimation
#     """

#     pos = np.zeros(signal.size, dtype=np.float64)
#     pos[-1] = initial_pos
#     for i in range(signal.size):
#         if pos[i - 1] > 1e-10:
#             if signal[i] <= -upper:
#                 pos[i] = -1.0
#             else:
#                 pos[i] = pos[i - 1]
#         elif pos[i - 1] < -1e-10:
#             if signal[i] >= upper:
#                 pos[i] = 1.0
#             else:
#                 pos[i] = pos[i - 1]
#         else:
#             if signal[i] >= upper:
#                 pos[i] = 1.0
#             elif signal[i] <= -upper:
#                 pos[i] = -1.0

#     return pos


def signal2position(signal, upper, lower):

    pos_numba = signal2positionNumba(signal.values, upper, lower)
    return pd.Series(pos_numba, index=signal.index)


def signal2position_Numba(signal, upper, lower):
    if len(signal) < 1:
        return np.array([])
    pos_numba = signal2positionNumba(signal.values, upper, lower)
    return pos_numba


def signal2position_groupby(signal, lower_p, upper_p):

    temp_res = signal.groupby(level=0).apply(signal2positionNumba_groupby_nonzero, lower_percentile=lower_p, upper_percentile=upper_p)
    result = pd.Series(np.concatenate(temp_res.values, axis=0), index=signal.index)
    return result


def signal2position_groupby_test(signal, upper, lower):

    temp_res = signal.groupby(level=0).apply(signal2position_Numba, upper=upper, lower=lower)
    result = pd.Series(np.concatenate(temp_res.values, axis=0), index=signal.index)
    return result



def compute_minute_pnl_from_pos(pos, backtest_contract_data, stk_indicator, spread_bid_ask_array, conservative_backtest=1,
                                order_jump=1):

    (minute_pnl, minute_pnl_TC, minute_pnl_long, minute_pnl_short,
     minute_pnl_long_TC, minute_pnl_short_TC, raw_ret, raw_ret_TC,
     raw_ret_long, raw_ret_long_TC, raw_ret_short, raw_ret_short_TC, 
     total_holding_time, total_trade_time, total_long_time, 
     total_short_time) = _compute_minute_pnl_from_pos_numba(pos=get_aligned_pos(pos, backtest_contract_data),
                                                            ask=backtest_contract_data['AskPrice1'].values,
                                                            bid=backtest_contract_data['BidPrice1'].values,
                                                            mid=backtest_contract_data['Mid'].values,
                                                            high=backtest_contract_data['HighLimit'].values,
                                                            low=backtest_contract_data['LowLimit'].values,
                                                            close=backtest_contract_data['Close'].values,
                                                            stk_indicator=stk_indicator,
                                                            spread_bid_ask_array=spread_bid_ask_array,
                                                            conservative_backtest=int(conservative_backtest),
                                                            order_jump=float(order_jump))

    # mean_holding_period = total_holding_time / total_trade_time

    # backtest_contract_data.to_csv('stock_data_debug.csv')
    # debug_df = pd.DataFrame({'minute_pnl':minute_pnl, 'minute_pnl_long':minute_pnl_long, 'minute_pnl_short':minute_pnl_short, 'raw_ret':raw_ret})
    # debug_df.to_hdf('F:/pnl_debug/db_hdf1.h5', key='debug_pnl')

    # return (pd.Series(minute_pnl, index=backtest_contract_data.index), pd.Series(minute_pnl_TC, index=backtest_contract_data.index),
    #         pd.Series(minute_pnl_long, index=backtest_contract_data.index), pd.Series(minute_pnl_short, index=backtest_contract_data.index),
    #         pd.Series(minute_pnl_long_TC, index=backtest_contract_data.index), pd.Series(minute_pnl_short_TC, index=backtest_contract_data.index),
    #         pd.Series(raw_ret, index=backtest_contract_data.index), pd.Series(raw_ret_TC, index=backtest_contract_data.index))
    return (pd.Series(raw_ret, index=backtest_contract_data.index), pd.Series(raw_ret_TC, index=backtest_contract_data.index),
            pd.Series(raw_ret_long, index=backtest_contract_data.index),pd.Series(raw_ret_long_TC, index=backtest_contract_data.index),
            pd.Series(raw_ret_short, index=backtest_contract_data.index),  pd.Series(raw_ret_short_TC, index=backtest_contract_data.index),
            total_holding_time, total_trade_time, total_long_time, total_short_time)



@jit(nopython=True, cache=True)
def _compute_minute_pnl_from_pos_numba(pos, ask, bid, mid, high, low, close, stk_indicator, spread_bid_ask_array,
                                       open_var_cost=0.0, open_fix_cost=0.0, close_var_cost=0.0, close_fix_cost=0.0,
                                       close_today_var_cost=0.0, close_today_fix_cost=0.0, date=None,
                                       conservative_backtest=0, order_jump=0):
    '''
    计算实际asset的变化

    :return: pnl 1d np.array 返回i~i+1 时刻的asset变化
    '''

    pnl = np.zeros((len(pos)))  # 初始化记录每分钟的不带手续费的pnl
    pnl_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    pnl_long = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    pnl_long_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    pnl_short = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    pnl_short_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret_long = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret_short = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret_long_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl
    raw_ret_short_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl

    jump = order_jump * spread_bid_ask_array * 0 # 实际挂单加减几跳
    target_pos = pos.copy()
    pos_sign = 0
    open_short_price = 1
    close_short_price = 1
    open_long_price = 1
    close_long_price = 1
    tc_open_long = 0
    tc_close_long = 0
    tc_open_short = 0
    tc_close_short = 0
    total_holding_time = 0
    open_pos_short = 0
    open_pos_long = 0
    total_trade_time = 0
    total_long_time = 0
    total_short_time = 0



    for i in range(len(pos)):

        if stk_indicator[i]:
            trade_postpone = False
            pos_sign = 0
            open_short_price = 1
            close_short_price = 1
            open_long_price = 1
            close_long_price = 1
            continue
        else:
            pos_change = pos[i] - pos[i - 1]  # 当前仓位的相对变化
            abs_pos_change = abs(pos[i]) - abs(pos[i - 1])  # 当前仓位的绝对值变化
            trade_postpone = False

            # 遇到涨跌停且和仓位变化方向一致，无法进入，强行让仓位变化为0
            if (ask[i] < 1 and pos_change > 0) or \
                    (bid[i] < 1 and pos_change < 0):
                trade_postpone = True
                target_pos[i] = target_pos[i - 1]
            # 只有当 买入挂单价格>下一分钟最低价时 或 卖出挂单价格<下一分钟最高价 才认为可以成交
            if conservative_backtest == 1:
                if (ask[i] + jump[i] <= low[i + 1] + 0.00001 and pos_change > 0) or \
                   (bid[i] - jump[i] >= high[i + 1] - 0.00001 and pos_change < 0):
                    trade_postpone = True
            # 只有当 买入挂单价格>下一分钟结束时的成交价 或 卖出挂单价格<下一分钟的成交价 才认为可以成交
            elif conservative_backtest == 2:
                if (ask[i] + jump[i] <= close[i + 1] + 0.00001 and pos_change > 0) or \
                   (bid[i] - jump[i] >= close[i + 1] - 0.00001 and pos_change < 0):
                    trade_postpone = True

            if trade_postpone:
                pos[i] = pos[i - 1]
                pos_change = 0.0
                abs_pos_change = 0.0

            # 不考虑任何交易成本，也考虑涨跌停无法交易的的情况下，pnl的变化
            if mid[i] != 0 and ask[i + 1] != 0 and bid[i + 1] != 0:
                pnl[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i]

                # 初始化考虑交易成本的pnl，这里已经考虑了挂单无法成交顺延下一分钟的情况
                pnl_TC[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i]

                pnl_long[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i] if target_pos[i] > 0 else 0
                pnl_short[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i] if target_pos[i] < 0 else 0
                pnl_long_TC[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i] if target_pos[i] > 0 else 0
                pnl_short_TC[i] = target_pos[i] * (mid[i + 1] - mid[i]) / mid[i] if target_pos[i] < 0 else 0

            # 在计算换仓手续费的时候，需要分四种情况：0->1，-1->0，1->0，0->-1
            if pos_change > 0:
                trade_price = ask[i] + jump[i]
                if abs_pos_change >= 0:  # 0 -> 1
                    pnl_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * open_var_cost + open_fix_cost) / trade_price
                    pnl_long_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * open_var_cost + open_fix_cost) / trade_price
                else:  # -1 -> 0
                    pnl_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * close_var_cost + close_fix_cost) / trade_price
                    pnl_short_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * close_var_cost + close_fix_cost) / trade_price
            elif pos_change < 0:
                trade_price = bid[i] - jump[i]
                if abs_pos_change > 0:  # 0 -> -1
                    pnl_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * open_var_cost + open_fix_cost) / trade_price
                    pnl_short_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * open_var_cost + open_fix_cost) / trade_price
                else:  # 1 -> 0
                    pnl_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * close_var_cost + close_fix_cost) / trade_price
                    pnl_long_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * close_var_cost + close_fix_cost) / trade_price

            if pos_change > 0:
                trade_price = ask[i] + jump[i]
                if abs_pos_change > 0 and pos_sign == 0:  # 0 -> 1
                    pos_sign = 1
                    tc_open_long = pos_change * (trade_price - mid[i] + trade_price * open_var_cost + open_fix_cost) / trade_price
                    open_long_price = mid[i]
                    open_pos_long = i

                elif abs_pos_change < 0 and pos_sign < 0:  # -1 -> 0
                    tc_close_short = pos_change * (trade_price - mid[i] + trade_price * close_var_cost + close_fix_cost) / trade_price
                    close_short_price = mid[i]
                    raw_ret[i] = (open_short_price - close_short_price) / open_short_price
                    raw_ret_short[i] = raw_ret[i]
                    raw_ret_TC[i] = raw_ret[i] - np.abs(tc_open_short) - np.abs(tc_close_short)
                    raw_ret_short_TC[i] = raw_ret_TC[i]
                    pos_sign = 0
                    total_holding_time += i - open_pos_short
                    total_trade_time += 1
                    total_short_time += 1

            elif pos_change < 0:
                trade_price = bid[i] - jump[i]
                if abs_pos_change > 0 and pos_sign == 0:  # 0 -> -1
                    pos_sign = -1
                    tc_open_short = pos_change * (mid[i] - trade_price + trade_price * open_var_cost + open_fix_cost) / trade_price
                    open_short_price = mid[i]
                    open_pos_short = i

                elif abs_pos_change < 0 and pos_sign > 0:  # 1 -> 0
                    tc_close_long = pos_change * (mid[i] - trade_price + trade_price * close_var_cost + close_fix_cost) / trade_price
                    close_long_price = mid[i]
                    raw_ret[i] = (close_long_price - open_long_price) / open_long_price
                    raw_ret_long[i] = raw_ret[i]
                    raw_ret_TC[i] = raw_ret[i] - np.abs(tc_open_long) - np.abs(tc_close_long)
                    raw_ret_long_TC[i] = raw_ret_TC[i]
                    pos_sign = 0
                    total_holding_time += i - open_pos_long
                    total_trade_time += 1
                    total_long_time += 1



    return pnl, pnl_TC, pnl_long, pnl_short, pnl_long_TC, pnl_short_TC, raw_ret, raw_ret_TC, raw_ret_long, raw_ret_long_TC, raw_ret_short, raw_ret_short_TC, total_holding_time, total_trade_time, total_long_time, total_short_time


def get_aligned_pos(pos, backtest_contract_data):

    return pos.reindex(index=backtest_contract_data.index, fill_value=0.0).values


def aggregate_to_daily_return(intraday_returns):
    """
    Aggregate an intraday return Pandas Series to
    a daily return Pandas Series.
    """
    return intraday_returns.groupby((intraday_returns.index).normalize()).aggregate(np.nansum)


def aggregate_to_min_return(intraday_returns, timeLevel):

    """
    aggregate ticker return into minute level
    to show the return details
    """
    return intraday_returns.resample(timeLevel, label='right').sum()


def split_datetime(date, time):
    return dt.datetime(date // 10000, date // 100 % 100, date % 100,
                       time // 10000000, time // 100000 % 100,
                       time // 1000 % 100)


def time_delta(timestamp, nperiod):

    delta = datetime.timedelta(seconds=3 * nperiod)
    return timestamp - delta


# @numba.jit(nopython=True)
# def metric_calculation(pos, stk_indicator, pnl_nocost_cumsum, pnl_cost_cumsum):
#     """
#     this funtion will calculate the average holding time for each transactions
#     """
#     pos_size = pos.size
#     i = 1
#     total_holding_time = 0.0
#     transaction_count = 0
#     win_count_nocost = 0
#     win_count_cost = 0
#     lose_count_nocost = 0
#     lose_count_cost = 0
#     win_return_nocost = 0.0
#     win_return_cost = 0.0
#     lose_return_nocost = 0.0
#     lose_return_cost = 0.0
#     trade_return_nocost = 0.0
#     trade_return_cost = 0.0
#     long_tran_count = 0
#     short_tran_count = 0
#     long_trade_return_nocost = 0.0
#     long_trade_return_cost = 0.0
#     short_trade_return_nocost = 0.0
#     short_trade_return_cost = 0.0
#     # initialization
#     if pos[0] != 0:
#         entry_index = 0
#     else:
#         entry_index = -1
#     while i < pos_size:
#         # open a position at time i
#         if stk_indicator[i]:
#             i += 1
#             continue
#         else:
#             if pos[i] != pos[i - 1]:
#                 if entry_index > 0:
#                     total_holding_time += i - entry_index
#                     transaction_count += 1

#                     if pos[i - 1] > 0: 
#                         if pos[i] == 0:
#                             long_tran_count += 1
#                             long_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                             long_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#                         else:
#                             long_tran_count += 1
#                             long_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index]
#                             long_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index]

#                     else:
#                         if pos[i] == 0:
#                             short_tran_count += 1
#                             short_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                             short_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#                         else:
#                             short_tran_count += 1
#                             short_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index]
#                             short_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index]


#                     trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                     trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]

#                     if pnl_nocost_cumsum[i] > pnl_nocost_cumsum[entry_index - 1]:
#                         win_count_nocost += 1
#                         win_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                     elif pnl_nocost_cumsum[i] < pnl_nocost_cumsum[entry_index - 1]:
#                         lose_count_nocost += 1
#                         lose_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]

#                     if pnl_cost_cumsum[i] > pnl_cost_cumsum[entry_index - 1]:
#                         win_count_cost += 1
#                         win_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#                     elif pnl_cost_cumsum[i] < pnl_cost_cumsum[entry_index - 1]:
#                         lose_count_cost += 1
#                         lose_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]

#                     if pos[i] != 0:
#                         entry_index = i
#                     else:
#                         entry_index = -1
#                 else:
#                     entry_index = i
#             # continue to next time step
#             i += 1
#     if entry_index > 0:
#         total_holding_time += i - entry_index
#         transaction_count += 1

#         if pos[i - 1] > 0:
#             if pos[i] == 0:
#                 long_tran_count += 1
#                 long_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                 long_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#             else:
#                 long_tran_count += 1
#                 long_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index]
#                 long_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index]

#         else:
#             if pos[i] == 0:
#                 short_tran_count += 1
#                 short_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#                 short_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#             else:
#                 short_tran_count += 1
#                 short_trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index]
#                 short_trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index]

#         trade_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#         trade_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]

#         if pnl_nocost_cumsum[i] > pnl_nocost_cumsum[entry_index - 1]:
#             win_count_nocost += 1
#             win_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]
#         elif pnl_nocost_cumsum[i] < pnl_nocost_cumsum[entry_index - 1]:
#             lose_count_nocost += 1
#             lose_return_nocost += pnl_nocost_cumsum[i] - pnl_nocost_cumsum[entry_index - 1]

#         if pnl_cost_cumsum[i] > pnl_cost_cumsum[entry_index - 1]:
#             win_count_cost += 1
#             win_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#         elif pnl_cost_cumsum[i] < pnl_cost_cumsum[entry_index - 1]:
#             lose_count_cost += 1
#             lose_return_cost += pnl_cost_cumsum[i] - pnl_cost_cumsum[entry_index - 1]
#     if transaction_count == 0:
#         transaction_count = 1e-8

#     mean_holding_time = total_holding_time / transaction_count

#     trade_mean_return_nocost = pnl_nocost_cumsum[-1] / transaction_count
#     trade_mean_return_cost = pnl_cost_cumsum[-1] / transaction_count

#     long_mean_return_nocost = long_trade_return_nocost / long_tran_count
#     long_mean_return_cost = long_trade_return_cost / long_tran_count

#     short_mean_return_nocost = short_trade_return_nocost / short_tran_count
#     short_mean_return_cost = short_trade_return_cost / short_tran_count

#     win_mean_return_nocost = win_return_nocost / win_count_nocost
#     win_mean_return_cost = win_return_cost / win_count_cost

#     lose_mean_return_nocost = lose_return_nocost / lose_count_nocost
#     lose_mean_return_cost = lose_return_cost / lose_count_cost

#     return (mean_holding_time, transaction_count,
#             long_tran_count, short_tran_count,
#             win_count_nocost, win_count_cost,
#             lose_count_nocost, lose_count_cost,
#             trade_mean_return_nocost, trade_mean_return_cost,
#             long_mean_return_nocost,  long_mean_return_cost, 
#             short_mean_return_nocost,  short_mean_return_cost,
#             win_mean_return_nocost,  win_mean_return_cost, 
#             lose_mean_return_nocost,  lose_mean_return_cost)


@numba.jit(nopython=True)
def metric_calculation(pos, stk_indicator, pnl_nocost_sum, pnl_cost_sum, pnl_long_sum_nocost,
                       pnl_short_sum_nocost, pnl_long_sum_cost, pnl_short_sum_cost):
    """
    this funtion will calculate the average holding time for each transactions
    """
    pos_size = pos.size
    i = 1
    total_holding_time = 0.0
    long_holding_time = 0.0
    short_holding_time = 0.0
    transaction_count = 0
    long_tran_count = 0
    short_tran_count = 0
    # initialization
    if pos[0] != 0:
        entry_index = 0
    else:
        entry_index = -1
    while i < pos_size:
        # open a position at time i
        if stk_indicator[i]:
            i += 1
            continue
        else:
            if pos[i] != pos[i - 1]:
                if entry_index > 0:
                    total_holding_time += i - entry_index
                    transaction_count += 1

                    if pos[i - 1] > 0: 
                        long_tran_count += 1
                        long_holding_time += i - entry_index

                    else:
                        short_tran_count += 1
                        short_holding_time += i - entry_index

                    if pos[i] != 0:
                        entry_index = i
                    else:
                        entry_index = -1
                else:
                    entry_index = i
            # continue to next time step
            i += 1
    if entry_index > 0:
        total_holding_time += i - entry_index
        transaction_count += 1

        if pos[i - 1] > 0:
            long_tran_count += 1
            long_holding_time += i - entry_index

        else:
            short_tran_count += 1
            short_holding_time += i - entry_index

    if transaction_count == 0:
        transaction_count = 1e-8

    if long_tran_count == 0:
        long_tran_count = 1e-8

    if short_tran_count == 0:
        short_tran_count = 1e-8

    mean_holding_time = total_holding_time / transaction_count
    mean_holding_time_long = long_holding_time / long_tran_count
    mean_holding_time_short = short_holding_time / short_tran_count

    trade_mean_return_nocost = pnl_nocost_sum / transaction_count
    trade_mean_return_cost = pnl_cost_sum / transaction_count

    long_mean_return_nocost = pnl_long_sum_nocost / long_tran_count
    long_mean_return_cost = pnl_long_sum_cost / long_tran_count

    short_mean_return_nocost = pnl_short_sum_nocost / short_tran_count
    short_mean_return_cost = pnl_short_sum_cost / short_tran_count

    return (mean_holding_time, transaction_count,
            long_tran_count, short_tran_count,
            mean_holding_time_long, mean_holding_time_short,
            trade_mean_return_nocost, trade_mean_return_cost,
            long_mean_return_nocost,  long_mean_return_cost, 
            short_mean_return_nocost,  short_mean_return_cost)


def metric_calculation_wrapper(pos, stk_indicator, pnl_nocost_sum, pnl_cost_sum, pnl_long_sum_nocost,
                               pnl_short_sum_nocost, pnl_long_sum_cost, pnl_short_sum_cost, raw_ret, raw_ret_TC):

    (mean_holding_time, transaction_count,
     long_tran_count, short_tran_count,
     mean_holding_time_long, mean_holding_time_short,
     trade_mean_return_nocost, trade_mean_return_cost,
     long_mean_return_nocost, long_mean_return_cost, 
     short_mean_return_nocost, short_mean_return_cost) = metric_calculation(pos, stk_indicator, pnl_nocost_sum, pnl_cost_sum, pnl_long_sum_nocost,
                                                                            pnl_short_sum_nocost, pnl_long_sum_cost, pnl_short_sum_cost)
    
    print('transaction_count:{}'.format(transaction_count))
    print('long_tran_count:{}'.format(long_tran_count))
    print('short_tran_count:{}'.format(short_tran_count))

    raw_ret_win_part = raw_ret[raw_ret > 0]
    # raw_ret_TC_win_part = raw_ret_TC[raw_ret_TC > 0]
    count_win_nocost = len(raw_ret_win_part)
    # count_win_cost = len(raw_ret_TC_win_part)

    mean_ret_win_nocost = raw_ret_win_part.mean() if count_win_nocost > 0 else 0
    # mean_ret_win_cost = raw_ret_TC_win_part.mean() if count_win_cost > 0 else 0

    raw_ret_lose_part = raw_ret[raw_ret < 0]
    # raw_ret_TC_lose_part = raw_ret_TC[raw_ret_TC < 0]
    count_lose_nocost = len(raw_ret_lose_part)
    # count_lose_cost = len(raw_ret_TC_lose_part)

    mean_ret_lose_nocost = raw_ret_lose_part.mean() if count_lose_nocost > 0 else 0
    # mean_ret_lose_cost = raw_ret_TC_lose_part.mean() if count_lose_cost > 0 else 0

    result = {"mean_holding_time": mean_holding_time, 
              # "mean_holding_time_long": mean_holding_time_long, 
              # "mean_holding_time_short": mean_holding_time_short,
              "count_trade": transaction_count,
              "count_long":long_tran_count, 
              "count_short":short_tran_count,
              "count_win_nocost": count_win_nocost,
              # "count_win_cost":count_win_cost,
              "count_lose_nocost": count_lose_nocost,
              # "count_lose_cost":count_lose_cost,
              "mean_ret_win_nocost": mean_ret_win_nocost,
              # "mean_ret_win_cost": mean_ret_win_cost,
              "mean_ret_lose_nocost": mean_ret_lose_nocost,
              # "mean_ret_lose_cost": mean_ret_lose_cost,
              "mean_return_nocost_all": trade_mean_return_nocost,
              "mean_return_cost_all": trade_mean_return_cost,
              "mean_return_nocost_long": long_mean_return_nocost,
              "mean_return_cost_long": long_mean_return_cost,
              "mean_return_nocost_short": short_mean_return_nocost, 
              "mean_return_cost_short": short_mean_return_cost}
    return result


def metric_calculation_wrapper_new(pos, backtest_contract_data, stk_indicator, spread_bid_ask_array):


    (raw_ret, raw_ret_TC, raw_ret_long, raw_ret_long_TC,
     raw_ret_short, raw_ret_short_TC, total_holding_time,
     total_trade_time, total_long_time, total_short_time) = compute_minute_pnl_from_pos(pos, backtest_contract_data, stk_indicator, spread_bid_ask_array, conservative_backtest=1,
                                                                                        order_jump=1)
    
    mean_holding_time = total_holding_time / total_trade_time if total_trade_time > 0 else 0
    trade_mean_return_nocost = raw_ret.sum() / total_trade_time if total_trade_time > 0 else 0
    trade_mean_return_cost = raw_ret_TC.sum() / total_trade_time if total_trade_time > 0 else 0
    long_mean_return_nocost = raw_ret_long.sum() / total_long_time if total_long_time > 0 else 0
    short_mean_return_nocost = raw_ret_short.sum() / total_short_time if total_short_time > 0 else 0
    long_mean_return_cost = raw_ret_long_TC.sum() / total_long_time if total_long_time > 0 else 0
    short_mean_return_cost = raw_ret_short_TC.sum() / total_long_time if total_short_time > 0 else 0

    raw_ret_win_part = raw_ret[raw_ret > 0]
    # raw_ret_TC_win_part = raw_ret_TC[raw_ret_TC > 0]
    count_win_nocost = len(raw_ret_win_part)
    # count_win_cost = len(raw_ret_TC_win_part)
    mean_ret_win_nocost = raw_ret_win_part.mean() if count_win_nocost > 0 else 0
    # mean_ret_win_cost = raw_ret_TC_win_part.mean() if count_win_cost > 0 else 0
    raw_ret_lose_part = raw_ret[raw_ret < 0]
    # raw_ret_TC_lose_part = raw_ret_TC[raw_ret_TC < 0]
    count_lose_nocost = len(raw_ret_lose_part)
    # count_lose_cost = len(raw_ret_TC_lose_part)
    mean_ret_lose_nocost = raw_ret_lose_part.mean() if count_lose_nocost > 0 else 0
    # mean_ret_lose_cost = raw_ret_TC_lose_part.mean() if count_lose_cost > 0 else 0

    result = {"mean_holding_time": mean_holding_time, 
              # "mean_holding_time_long": mean_holding_time_long, 
              # "mean_holding_time_short": mean_holding_time_short,
              "count_trade": total_trade_time,
              "count_long":total_long_time, 
              "count_short":total_short_time,
              "count_win_nocost": count_win_nocost,
              # "count_win_cost":count_win_cost,
              "count_lose_nocost": count_lose_nocost,
              # "count_lose_cost":count_lose_cost,
              "mean_ret_win_nocost": mean_ret_win_nocost,
              # "mean_ret_win_cost": mean_ret_win_cost,
              "mean_ret_lose_nocost": mean_ret_lose_nocost,
              # "mean_ret_lose_cost": mean_ret_lose_cost,
              "mean_return_nocost_all": trade_mean_return_nocost,
              "mean_return_cost_all": trade_mean_return_cost,
              "mean_return_nocost_long": long_mean_return_nocost,
              "mean_return_cost_long": long_mean_return_cost,
              "mean_return_nocost_short": short_mean_return_nocost, 
              "mean_return_cost_short": short_mean_return_cost}
    return (raw_ret, raw_ret_TC, result)



@numba.jit(nopython=True)
def trading_times_stat(pos, stk_indicator):
    """
    this funtion will calculate the total trading times for the trading strategy.
    """
    pos_size = pos.size
    i = 1
    transaction_count = 0
    # initialization
    if pos[0] != 0:
        entry_index = 0
    else:
        entry_index = -1
    while i < pos_size:
        # open a position at time i
        if stk_indicator[i]:
            i += 1
            continue
        else:
            if pos[i] != pos[i - 1]:
                if entry_index > 0:
                    transaction_count += 1
                    if pos[i] != 0:
                        entry_index = i
                    else:
                        entry_index = -1
                else:
                    entry_index = i
            # continue to next time step
            i += 1
    if entry_index > 0:
        transaction_count += 1
    return transaction_count





@numba.jit(nopython=True)
def sharpe(daily_returns, trading_days=252.0, annulized_rf=0.):
    """
    Compute the sharpe ratio for given daily return time series.
    """
    return_mean = np.nanmean(daily_returns)
    return_mean2 = np.nanmean(daily_returns * daily_returns)
    return_var = return_mean2 - return_mean * return_mean
    if return_var == 0:
        return 0.0
    elif return_var < 0:
        return_var = return_mean2

    return (return_mean - annulized_rf / trading_days) \
        / np.sqrt(return_var / trading_days)


@numba.jit(nopython=True)
def sharpe_min_level(min_returns, mins, trading_days=252.0, annulized_rf=0.):
    """
    Compute the sharpe ratio for given daily return time series.
    """
    return_mean = np.nanmean(min_returns)
    return_mean2 = np.nanmean(min_returns * min_returns)
    return_var = return_mean2 - return_mean * return_mean
    if return_var == 0:
        return 0.0
    elif return_var < 0:
        return_var = return_mean2
    return (return_mean - annulized_rf / (trading_days * 240 / mins)) / np.sqrt(return_var / (trading_days * 240 / mins))


@numba.jit(nopython=True)
def mdd_ret_min_level(returns, mins, trading_days=252.0):
    """
    Compute the maxdrawdown given daily return time series.
    """
    max_asset = 0.0
    max_drawdown = -1.0
    current_asset = 0.0
    for i in range(len(returns)):
        current_asset += returns[i]
        max_asset = max(max_asset, current_asset)
        max_drawdown = max(max_drawdown, max_asset - current_asset)
    mean_ret = current_asset / len(returns)
    if max_drawdown <= 0:
        # return -1, -1
        return 0
    else:
        # return mean_ret * trading_days * 240 / mins, max_drawdown
        return max_drawdown


def back_test_old(featureInput, stock, stk_indicator, spread_bid_ask_array, upbound, lowbound):

    # ===================================
    # lower, upper = transform(featureInput, upbound, lowbound)
    # # pos = signal2position(featureInput, lower, upper)
    # pos = signal2position_groupby_test(featureInput, upper, lower)
    # =================================== old way
    pos = signal2position_groupby(featureInput, lower_p=lowbound, upper_p=upbound)
    # mean_holding_period = mean_holding_time(pos.values, stk_indicator)
    pnl_NoCost, pnl_Cost, pnl_long, pnl_short, pnl_long_TC, pnl_short_TC, raw_ret, raw_ret_TC= compute_minute_pnl_from_pos(pos, stock, stk_indicator, spread_bid_ask_array)

    pnl_nocost_sum = np.sum(np.where(np.isfinite(pnl_NoCost.values), pnl_NoCost.values, 0))
    pnl_cost_sum = np.sum(np.where(np.isfinite(pnl_Cost.values), pnl_Cost.values, 0))

    pnl_long_nocost_sum = np.sum(np.where(np.isfinite(pnl_long.values), pnl_long.values, 0))
    pnl_long_cost_sum = np.sum(np.where(np.isfinite(pnl_long_TC.values), pnl_long_TC.values, 0))
    pnl_short_nocost_sum = np.sum(np.where(np.isfinite(pnl_short.values), pnl_short.values, 0))
    pnl_short_cost_sum = np.sum(np.where(np.isfinite(pnl_short_TC.values), pnl_short_TC.values, 0))

    # pnl_nocost_cumsum = np.cumsum(np.where(np.isfinite(pnl_NoCost.values), pnl_NoCost.values, 0))
    # pnl_cost_cumsum = np.cumsum(np.where(np.isfinite(pnl_Cost.values), pnl_Cost.values, 0))

    # de_bug = pd.DataFrame({'pos':pos.values, 'pnl_nocost_cumsum':pnl_nocost_cumsum, 'pnl_cost_cumsum':pnl_cost_cumsum}, index=pos.index)
    # de_bug.to_csv('debug_pos_cumsum.csv')
    metric_result = metric_calculation_wrapper(pos.values, stk_indicator, pnl_nocost_sum, pnl_cost_sum, pnl_long_nocost_sum, pnl_short_nocost_sum,
                                               pnl_long_cost_sum, pnl_short_cost_sum, raw_ret.values, raw_ret_TC.values)

    return pnl_NoCost, pnl_Cost, metric_result


def back_test(featureInput, stock, stk_indicator, spread_bid_ask_array, upbound, lowbound):

    # ===================================
    # lower, upper = transform(featureInput, upbound, lowbound)
    # # pos = signal2position(featureInput, lower, upper)
    # pos = signal2position_groupby_test(featureInput, upper, lower)
    # =================================== old way
    pos = signal2position_groupby(featureInput, lower_p=lowbound, upper_p=upbound)
    raw_ret, raw_ret_TC, metric_result = metric_calculation_wrapper_new(pos, stock, stk_indicator, spread_bid_ask_array)

    return raw_ret, raw_ret_TC, metric_result


def pnl_result_plot(pnl_sum, feature_name):

    plt.figure(figsize=(12, 9), dpi=120)
    plt.plot(pnl_sum)
    plt.legend(pnl_sum.columns)
    plt.title(feature_name)
    plt.savefig(feature_name + '.png')
    plt.clf()
