import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import auxiliaryFunctionsNew1009 as af
import gc
import sys
from scipy import stats
import numba
from numba import jit


def backtest_check(params_dict):
    csv_res = dict()
    feature_hd = pd.HDFStore(params_dict['h5_result_name'], 'r')
    target_hd = pd.HDFStore(params_dict['h5_target_name'], 'r')
    feature_data = feature_hd[params_dict['feature_col']]
    pnl_data_concated = target_hd['pnl_data_concated']
    spread_bid_ask_series = target_hd['spread_bid_ask_series']

    select_intersect = np.isfinite(feature_data).values & np.isfinite(pnl_data_concated).all(axis=1).values & np.isfinite(spread_bid_ask_series).values

    stk = pnl_data_concated.loc[select_intersect]
    signal_pnl = feature_data.loc[select_intersect]
    spread_bid_ask_array = spread_bid_ask_series.loc[select_intersect].values

    stk_index_label = pd.Index(stk.index.labels[0])
    stk_indicator = stk_index_label.duplicated(keep='first') ^ stk_index_label.duplicated(keep='last')  # sign for the beginning and the end of each ticker

    pnl_dict_nocost, pnl_dict_cost = dict(), dict()
    u, l = params_dict['up_low']
    pnl_dict_nocost['up_{}_low_{}'.format(u, l)], pnl_dict_cost['up_{}_low_{}'.format(u, l)], csv_res = af.back_test(signal_pnl, stk, stk_indicator, spread_bid_ask_array, u, l)
    gc.collect()
    del gc.garbage[:]
    gc.collect()
    pnl_nocost_concated = pd.concat(pnl_dict_nocost, axis=1, join='outer')
    pnl_cost_concated = pd.concat(pnl_dict_cost, axis=1, join='outer')
    pnl_nocost_no_na = pnl_nocost_concated.dropna()
    pnl_cost_no_na = pnl_cost_concated.dropna()

    return pnl_nocost_no_na, pnl_cost_no_na


def get_daily_returns(pnl_cleaned):

    new_pnl_cleaned = pd.Series(pnl_cleaned.values.flatten(), index=pnl_cleaned.index)

    def aggregate_to_daily_return(intraday_returns):

    # Aggregate an intraday return Pandas Series to
    # a daily return Pandas Series.

        return intraday_returns.groupby((intraday_returns.index).normalize()).aggregate(np.nansum)

    pnl_unstacked = new_pnl_cleaned.unstack(0)
    result = pnl_unstacked.groupby(pnl_unstacked.columns, axis=1).apply(aggregate_to_daily_return)  # sum each ticker into daily level
    result.columns = result.columns.droplevel(1)  # remove one level of the columns
    return result


def from_tick_to_daily(tick_path, ticker_list, save_path, start_time, end_time):

    tick_hd = pd.HDFStore(tick_path, 'r')
    status_col = "columns ={}".format(['Close', 'TransactionNum', 'TransactionVol', 'TransactionAmount'])
    index_select = "index>pd.Timestamp(\'{}\') & index<pd.Timestamp(\'{}\')".format(start_time, end_time)
    for ticker in ticker_list:
        data_ticker = tick_hd.select(ticker, index_select + "&" + status_col)
        resample_tickers_to_daily(data_ticker, save_path, ticker)
    tick_hd.close()


def resample_tickers_to_daily(ticker_data, save_path, ticker_name):

    ohlc_ = ticker_data['Close'].resample('D', closed='right', label='right').ohlc()
    tran_ = ticker_data[['TransactionNum', 'TransactionVol', 'TransactionAmount']].resample('D', closed='right', label='right').sum()
    overall_ = pd.concat((ohlc_, tran_), axis=1).dropna()
    overall_.to_hdf(save_path, key=ticker_name, format='fixed')


def concat_columns_betweem_tickers(ticker_list, column_name, ticker_daily_path):

    ticker_daily_hd = pd.HDFStore(ticker_daily_path, 'r')
    result_dict = dict()
    for ticker in ticker_list:
        result_dict[ticker] = ticker_daily_hd[ticker][column_name]
    ticker_daily_hd.close()   
    result = pd.concat(result_dict, axis=1, keys=ticker_list, join='outer')
    return result


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

def signal2position(signal, upper, lower):

    pos_numba = signal2positionNumba(signal.values, upper, lower)
    return pd.Series(pos_numba, index=signal.index)


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
    raw_ret_TC = np.zeros((len(pos)))  # 初始化记录每分钟带手续费的pnl

    jump = order_jump * spread_bid_ask_array * 0.5 # 实际挂单加减几跳
    target_pos = pos.copy()
    pos_sign = 0
    open_long_prices = np.zeros((len(pos)))
    close_long_prices = np.zeros((len(pos)))
    open_short_prices = np.zeros((len(pos)))
    close_short_prices = np.zeros((len(pos)))
    open_short_price = 1
    close_short_price = 1
    open_long_price = 1
    close_long_price = 1
    tc_open_long = 0
    tc_close_long = 0
    tc_open_short = 0
    tc_close_short = 0



    for i in range(len(pos)):

        if stk_indicator[i]:
            trade_postpone = False
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
                    open_long_prices[i] = trade_price
                    pnl_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * open_var_cost + open_fix_cost) / trade_price
                    pnl_long_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * open_var_cost + open_fix_cost) / trade_price
                else:  # -1 -> 0
                    close_short_prices[i] = trade_price
                    pnl_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * close_var_cost + close_fix_cost) / trade_price
                    pnl_short_TC[i] -= pos_change * (trade_price - mid[i] + trade_price * close_var_cost + close_fix_cost) / trade_price
            elif pos_change < 0:
                trade_price = bid[i] - jump[i]
                if abs_pos_change > 0:  # 0 -> -1
                    open_short_prices[i] = trade_price
                    pnl_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * open_var_cost + open_fix_cost) / trade_price
                    pnl_short_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * open_var_cost + open_fix_cost) / trade_price
                else:  # 1 -> 0
                    close_long_prices[i] = trade_price
                    pnl_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * close_var_cost + close_fix_cost) / trade_price
                    pnl_long_TC[i] -= -pos_change * (mid[i] - trade_price + trade_price * close_var_cost + close_fix_cost) / trade_price

    return pnl_TC, pnl_long_TC, pnl_short_TC, open_long_prices, close_short_prices, open_short_prices, close_long_prices


def compute_minute_pnl_from_pos(pos, backtest_contract_data, stk_indicator, spread_bid_ask_array, conservative_backtest=1,
                                order_jump=1):

    (minute_pnl_TC, minute_pnl_long_TC, 
     minute_pnl_short_TC, open_long_prices, 
     close_short_prices, open_short_prices, close_long_prices) = _compute_minute_pnl_from_pos_numba(pos=get_aligned_pos(pos, backtest_contract_data),
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

    return (pd.Series(minute_pnl_TC, index=backtest_contract_data.index),
            pd.Series(minute_pnl_long_TC, index=backtest_contract_data.index), 
            pd.Series(minute_pnl_short_TC, index=backtest_contract_data.index),
            pd.Series(open_long_prices, index=backtest_contract_data.index),
            pd.Series(close_short_prices, index=backtest_contract_data.index), 
            pd.Series(open_short_prices, index=backtest_contract_data.index),
            pd.Series(close_long_prices, index=backtest_contract_data.index))













if __name__ == '__main__':
    params_dict = dict()
    params_dict['h5_result_name'] = 'F:/TOPT_1029/modify_feature/Ask_Bid_1_New.h5'
    params_dict['h5_target_name'] = 'F:/Machine_Learning_Structure/feature_files_old/Ask_Bid_1_New_target.h5'
    params_dict['feature_col'] = 'nperiod:10'
