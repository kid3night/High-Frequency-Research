# define BacktestEngine class
# import sys
# sys.path.append('./backtest_main_files/')
# print(sys.path)
import os
import pandas as pd
import numpy as np
import auxiliaryFunctionsNew1009 as af
from datafeed import dataFeed
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp
import importlib
import itertools as it
import scipy.stats as sts
import warnings
import tables
import backtest_engine_support as bes
import gc
warnings.simplefilter(action='ignore', category=tables.NaturalNameWarning)


class backtest_engine:

    def __init__(self, start_time, end_time, output_path, feature_name, feature_type, each_type_data_need, no_pnl=0, multi_proc_feature=0, multi_proc_result=0, 
                 upper=[90], lower=[70], time_level='10min'):

        self.start_time = start_time
        self.end_time = end_time
        self.upper = upper
        self.lower = lower
        self.no_pnl = no_pnl
        self.time_level = time_level
        self.output_path = output_path
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.each_type_data_need = each_type_data_need
        self.multi_proc_feature = multi_proc_feature
        self.multi_proc_result = multi_proc_result
        self.tick_path = 'F:/tick_concated/tick_concated.h5'
        self.target_path = 'F:/targets_concated_temp/targets_concated.h5'
        self.order_path = 'F:/order_concated/order_concated.h5'
        self.transaction_path = 'F:/transaction_concated/transaction_concated.h5'
        self.orderbook_path = 'F:/order_concated_data_check/orderbook_concated.h5'
        self.target_scope = ['MidReturn15', 'MidReturn60', 'MidReturn120', 'MidReturn180',
                             'MidReturn300', 'Diff_Return_60-15', 'Diff_Return_120-60',
                             'Diff_Return_180-120', 'Diff_Return_300-180']
        self.up_low_comb = list(it.product(self.upper, self.lower))


    def get_feature_obj(self, feature_file_name, params):

        print('initialize feature...')
        sf = importlib.import_module('feature_files.' + feature_file_name)
        self.feature_obj = getattr(sf, self.feature_name)(self.feature_name, self.feature_type, params)
        print('parameters include: {}'.format(self.feature_obj.params_comb_names))


    def cal_feature_target(self, ticker):

        data_fed = dataFeed(self.start_time, self.end_time, ticker, self.tick_path, self.target_path,
                            self.order_path, self.transaction_path, self.orderbook_path, self.feature_type,
                            self.each_type_data_need)
        result_list = [self.feature_obj.compute_feature(data_fed.data_feed, data_fed.data_affiliated['status']),
                       data_fed.data_affiliated['target_data'],
                       data_fed.data_affiliated['level1_data']]
        gc.collect()
        del gc.garbage[:]
        gc.collect()
        return result_list

    def multi_process_calculate(self, universe):

        p = mp.Pool(self.multi_proc_feature)
        results_tuple = p.map(self.cal_feature_target, iterable=universe)
        p.close()
        p.join()
        return results_tuple


    def feature_target_pair_generator(self, universe):

        print('prepare data and calculate feature...')
        feature_dict , target_dict, pnl_data_dict, ticker_list = dict(), dict(), dict(), list()
        if not self.multi_proc_feature:
            for ticker in universe:
                feature_signal, target_data, pnl_data = self.cal_feature_target(ticker)
                if feature_signal is not None and not feature_signal.empty:
                    feature_dict[ticker] = feature_signal
                    target_dict[ticker] = target_data
                    pnl_data_dict[ticker] = pnl_data
                    ticker_list.append(ticker)

        elif self.multi_proc_feature:
            temp_result_tuple = self.multi_process_calculate(universe)
            for ix, sub_tuple in enumerate(temp_result_tuple):
                if sub_tuple[0] is not None and not sub_tuple[0].empty:
                    feature_dict[universe[ix]] = sub_tuple[0]
                    target_dict[universe[ix]] = sub_tuple[1]
                    pnl_data_dict[universe[ix]] = sub_tuple[2]
                    ticker_list.append(universe[ix])
        self.feature_concated, self.target_concated, self.pnl_data_concated, self.ticker_list = pd.concat(feature_dict, axis=0, keys=ticker_list), pd.concat(target_dict, axis=0, keys=ticker_list), pd.concat(pnl_data_dict, axis=0, keys=ticker_list), ticker_list
        del feature_dict, target_dict, pnl_data_dict
        gc.collect()
        del gc.garbage[:]
        gc.collect()


    def create_rout(self, universe_name):

        print('create output route..')

        if self.feature_concated.empty:
            print('No signal generated !!')
        else:
            folder_name = 'Test_{}_On_{}_From_{}_To_{}'.format(self.feature_name, universe_name, self.start_time, self.end_time)
            self.output_rout = self.output_path + '/' + folder_name
            if not os.path.exists(self.output_rout):
                os.makedirs(self.output_rout)
            self.h5_result_name = self.output_rout + '/' + self.feature_name + '.h5'
            self.h5_target_name = self.output_rout + '/' + self.feature_name + '_target' + '.h5'
            other_para_dict = {'feature_length':len(self.feature_concated), 'target_length':len(self.target_concated), 'pnl_length':len(self.pnl_data_concated)}
            other_para_series = pd.Series(other_para_dict)
            spread_bid_ask = (self.pnl_data_concated['AskPrice1'] - self.pnl_data_concated['BidPrice1']).groupby(level=0).apply(lambda spread: np.min(spread[spread > 0])).fillna(0)
            self.spread_bid_ask_series = spread_bid_ask.reindex(self.pnl_data_concated.index.get_level_values(0))
            for col in self.feature_concated.columns:
                self.feature_concated[col].to_hdf(self.h5_result_name, key=col, mode='a')
            self.feature_concated.describe().to_hdf(self.h5_result_name, key='feature_describe', mode='a')
            ticker_series = pd.Series(self.ticker_list)
            ticker_series.to_hdf(self.h5_result_name, key='ticker_list', mode='a')
            other_para_series.to_hdf(self.h5_result_name, key='other_paras', mode='a')
            for col in self.target_concated.columns:
                self.target_concated[col].to_hdf(self.h5_target_name, key=col, mode='a')
            # if not self.no_pnl:
            self.pnl_data_concated.to_hdf(self.h5_target_name, key='pnl_data_concated', mode='a')
            self.spread_bid_ask_series.to_hdf(self.h5_target_name, key='spread_bid_ask_series', mode='a')
            print('data saveing finished...')
            del self.target_concated, self.feature_concated, self.pnl_data_concated, self.spread_bid_ask_series, spread_bid_ask
            gc.collect()
            del gc.garbage[:]
            gc.collect()


    def read_from_data_backtest(self, feature_file_name, params, saved_data_path, new_result_folder_name, universe_name):

        print('read from generated feature data...')

        self.get_feature_obj(feature_file_name, params)
        folder_name = 'Test_{}_On_{}_From_{}_To_{}'.format(self.feature_name, universe_name, self.start_time, self.end_time)
        self.h5_result_name = saved_data_path + '/' + self.feature_name + '.h5'
        self.h5_target_name = saved_data_path + '/' + self.feature_name + '_target' + '.h5'
        self.output_rout = self.output_path + '/' + folder_name + '/' + new_result_folder_name
        if not os.path.exists(self.output_rout):
            os.makedirs(self.output_rout)
        self.back_test()


    def back_test(self):

        if self.no_pnl == 1:
            self.feature_backtest_without_pnl()
        elif self.no_pnl == 0:
            self.feature_backtest_overall()
        else:
            self.feature_backtest_pnl_only()


    def feature_backtest(self):

        other_paras = pd.read_hdf(self.h5_result_name, key='other_paras')
        assert other_paras['feature_length'] == other_paras['target_length'] and other_paras['feature_length'] == other_paras['pnl_length']
        spread_bid_ask_series = pd.read_hdf(self.h5_target_name, key='spread_bid_ask_series')
        self.spread_bid_ask_array = spread_bid_ask_series.values
        if not self.multi_proc_result:
            print('single process backtesting...')
            result = dict()
            pnl_overall_nocost = dict()
            pnl_overall_cost = dict()
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]
            params = dict(zip(keys, data))
            for i in range(len(params['feature_columns'])):
                pnl_overall_nocost[self.feature_obj.feature_columns[i]], pnl_overall_cost[self.feature_obj.feature_columns[i]], result[self.feature_obj.feature_columns[i]], _ = bes.back_test_inside_functon_outside(i=i, params_dict=params)
            pnl_nc = pd.concat(pnl_overall_nocost, axis=1, keys=self.feature_obj.feature_columns, join='outer')
            pnl_c = pd.concat(pnl_overall_cost, axis=1, keys=self.feature_obj.feature_columns, join='outer')
            out_put_result = pd.DataFrame(result)
        else:
            print('multiprocessing testing...')

            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]
            params = dict(zip(keys, data))
            p = mp.Pool(self.multi_proc_result)
            results_tuple = p.map(partial(bes.back_test_inside_functon_outside, params_dict=params), iterable=range(len(params['feature_columns'])))
            results_tuple = list(zip(*results_tuple))
            p.close()
            p.join()
            pnl_nc = pd.concat(results_tuple[0], axis=1, keys=results_tuple[3], join='outer')
            pnl_c = pd.concat(results_tuple[1], axis=1, keys=results_tuple[3], join='outer')
            out_put_result = pd.DataFrame(list(results_tuple[2]), index=results_tuple[3]).T
        pnl_nc.fillna(0).cumsum().plot()
        plt.suptitle('Pnl_Plot_With_No_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_no_cost.png')
        pnl_c.fillna(0).cumsum().plot()
        plt.suptitle('Pnl_Plot_With_No_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_with_cost.png')
        out_put_result.to_csv(self.output_rout + '/' + 'back_test.csv')


    def feature_backtest_without_pnl(self):

        other_paras = pd.read_hdf(self.h5_result_name, key='other_paras')
        assert other_paras['feature_length'] == other_paras['target_length'] and other_paras['feature_length'] == other_paras['pnl_length']
        if not self.multi_proc_result:
            print('single process backtesting...')
            result = dict()
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'up_low_comb',
                    'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.up_low_comb, self.time_level]
            params = dict(zip(keys, data))
            for i in range(len(params['feature_columns'])):
                result[self.feature_obj.feature_columns[i]], _ = bes.back_test_inside_functon_outside_without_pnl(i=i, params_dict=params)
            out_put_result = pd.DataFrame(result)
        else:
            print('multiprocessing testing...')
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'up_low_comb',
                    'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.up_low_comb, self.time_level]
            params = dict(zip(keys, data))
            p = mp.Pool(self.multi_proc_result)
            results_tuple = p.map(partial(bes.back_test_inside_functon_outside_without_pnl, params_dict=params), iterable=range(len(params['feature_columns'])))
            results_tuple = list(zip(*results_tuple))
            p.close()
            p.join()
            out_put_result = pd.DataFrame(list(results_tuple[0]), index=results_tuple[1]).T
        out_put_result.to_csv(self.output_rout + '/' + 'back_test.csv')



    def feature_backtest_overall(self):

        other_paras = pd.read_hdf(self.h5_result_name, key='other_paras')
        assert other_paras['feature_length'] == other_paras['target_length'] and other_paras['feature_length'] == other_paras['pnl_length']
        spread_bid_ask_series = pd.read_hdf(self.h5_target_name, key='spread_bid_ask_series')
        self.spread_bid_ask_array = spread_bid_ask_series.values
        if not self.multi_proc_result:
            print('single process backtesting...')
            result = dict()
            result_pnl = dict()
            pnl_overall_nocost = dict()
            pnl_overall_cost = dict()
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]

            params = dict(zip(keys, data))
            for i in range(len(params['feature_columns'])):
                result[self.feature_obj.feature_columns[i]], _ = bes.back_test_inside_functon_outside_without_pnl(i=i, params_dict=params)
                pnl_overall_nocost[self.feature_obj.feature_columns[i]], pnl_overall_cost[self.feature_obj.feature_columns[i]], result_pnl[self.feature_obj.feature_columns[i]], _ = bes.backtest_pnl_info_part(i=i, params_dict=params)

            out_put_result = pd.DataFrame(result)
            out_put_result_pnl = pd.DataFrame(result_pnl)
            pnl_nc = pd.concat(pnl_overall_nocost, axis=1, join='outer')
            pnl_c = pd.concat(pnl_overall_cost, axis=1, join='outer')
        else:
            print('multiprocessing testing...')
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]


            params = dict(zip(keys, data))
            p = mp.Pool(self.multi_proc_result)
            results_tuple = p.map(partial(bes.back_test_inside_functon_outside_without_pnl, params_dict=params), iterable=range(len(params['feature_columns'])))
            results_tuple = list(zip(*results_tuple))
            results_tuple_pnl = p.map(partial(bes.backtest_pnl_info_part, params_dict=params), iterable=range(len(params['feature_columns'])))
            results_tuple_pnl = list(zip(*results_tuple_pnl))
            p.close()
            p.join()
            out_put_result = pd.DataFrame(list(results_tuple[0]), index=results_tuple[1]).T
            out_put_result_pnl = pd.DataFrame(list(results_tuple_pnl[2]), index=results_tuple_pnl[3]).T

            pnl_nc = pd.concat(results_tuple_pnl[0], axis=1, keys=results_tuple_pnl[3], join='outer')
            pnl_c = pd.concat(results_tuple_pnl[1], axis=1, keys=results_tuple_pnl[3], join='outer')


        out_put_result.to_csv(self.output_rout + '/' + 'back_test.csv')
        out_put_result_pnl.to_csv(self.output_rout + '/' + 'back_test_pnl.csv')
        pnl_nc.fillna(0).cumsum().plot()
        plt.suptitle('Pnl_Plot_With_No_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_no_cost.png')
        pnl_c.fillna(0).cumsum().plot()
        plt.suptitle('Pnl_Plot_With_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_with_cost.png')
        

    def feature_backtest_pnl_only(self):

        other_paras = pd.read_hdf(self.h5_result_name, key='other_paras')
        assert other_paras['feature_length'] == other_paras['target_length'] and other_paras['feature_length'] == other_paras['pnl_length']
        spread_bid_ask_series = pd.read_hdf(self.h5_target_name, key='spread_bid_ask_series')
        self.spread_bid_ask_array = spread_bid_ask_series.values
        if not self.multi_proc_result:
            print('single process backtesting...')
            result = dict()
            result_pnl = dict()
            pnl_overall_nocost = dict()
            pnl_overall_cost = dict()
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]

            params = dict(zip(keys, data))
            for i in range(len(params['feature_columns'])):
                result_pnl[self.feature_obj.feature_columns[i]] = bes.backtest_pnl_info_part(i=i, params_dict=params)

            out_put_result_pnl = pd.DataFrame(result_pnl)
            pnl_nc = pd.concat(pnl_overall_nocost, axis=1, join='outer')
            pnl_c = pd.concat(pnl_overall_cost, axis=1, join='outer')
        else:
            print('multiprocessing testing...')
            keys = ['feature_columns', 'target_scope', 'output_rout',
                    'h5_result_name', 'h5_target_name', 'spread_bid_ask_array',
                    'up_low_comb', 'time_level']
            data = [self.feature_obj.feature_columns, self.target_scope, self.output_rout,
                    self.h5_result_name, self.h5_target_name, self.spread_bid_ask_array,
                    self.up_low_comb, self.time_level]


            params = dict(zip(keys, data))
            p = mp.Pool(self.multi_proc_result)
            results_tuple_pnl = p.map(partial(bes.backtest_pnl_info_part, params_dict=params), iterable=range(len(params['feature_columns'])))
            results_tuple_pnl = list(zip(*results_tuple_pnl))
            p.close()
            p.join()
            out_put_result_pnl = pd.DataFrame(list(results_tuple_pnl[2]), index=results_tuple_pnl[3]).T

            pnl_nc = pd.concat(results_tuple_pnl[0], axis=1, keys=results_tuple_pnl[3], join='outer')
            pnl_c = pd.concat(results_tuple_pnl[1], axis=1, keys=results_tuple_pnl[3], join='outer')


        out_put_result_pnl.to_csv(self.output_rout + '/' + 'back_test_pnl.csv')
        pnl_nc.fillna(0).cumsum().plot(figsize=(12, 9))
        plt.suptitle('Pnl_Plot_With_No_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_no_cost.png')
        pnl_c.fillna(0).cumsum().plot(figsize=(12, 9))
        plt.suptitle('Pnl_Plot_With_Cost')
        plt.savefig(self.output_rout + '/' + 'pnl_with_cost.png')