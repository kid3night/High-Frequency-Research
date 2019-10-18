# import sys
# sys.path.append('./backtest_main_files/')
# print(sys.path)
import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import auxiliaryFunctionsNew1009 as af
import gc
import sys
from scipy import stats


def back_test_inside_functon_outside(i, params_dict):

    print('testing parameter:{}...'.format(params_dict['feature_columns'][i]))
    csv_res = dict()
    # fig, ax = plt.subplots(3, 3, figsize=(12, 9), dpi=120)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig2, ax2 = plt.subplots(3, 3, figsize=(12, 9), dpi=120)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    feature_hd = pd.HDFStore(params_dict['h5_result_name'], 'r')
    target_hd = pd.HDFStore(params_dict['h5_target_name'], 'r')
    feature_data = feature_hd[params_dict['feature_columns'][i]]
    feature_isfinite = np.isfinite(feature_data).values
    pnl_data_concated = target_hd['pnl_data_concated']
    for j, target_name in enumerate(params_dict['target_scope']):
        target_raw = target_hd[target_name]
        target_isfinite = np.isfinite(target_raw).values
        intersct_vector = feature_isfinite & target_isfinite
        signal = feature_data.loc[intersct_vector]
        targt = target_raw.loc[intersct_vector]
        stk = pnl_data_concated.loc[intersct_vector, :]
        corr_pearson = round(np.corrcoef(signal, targt)[0][1], 3)
        corr_spearman = round(sts.spearmanr(signal, targt)[0], 3)
        csv_res[params_dict['target_scope'][j] + '_pearson_corr'] = corr_pearson
        csv_res[params_dict['target_scope'][j] + '_spearman_corr'] = corr_spearman
        # ret_sorted = af.return_sort_single(signal, targt)
        # ax[j // 3, j % 3].plot(np.arange(len(signal)), ret_sorted.cumsum())
        # ax[j // 3, j % 3].set_title(params_dict['target_scope'][j])
        return_percent = af.percentile_stats(signal, targt, 2, 2)
        ax2[j // 3, j % 3].bar(return_percent.index, return_percent.values.T[0])
        ax2[j // 3, j % 3].set_title(params_dict['target_scope'][j])
        del target_raw, target_isfinite, intersct_vector, signal, targt
        gc.collect()
        del gc.garbage[:]
        gc.collect()

    picure_name = params_dict['feature_columns'][i].replace(':', '_')
    fig.suptitle(picure_name + '_correlation_plot')
    fig.savefig(params_dict['output_rout'] + '/' + picure_name + '_correlation_plot.png')
    fig2.suptitle(picure_name + '_return_percent')
    fig2.savefig(params_dict['output_rout'] + '/' + picure_name + '_return_percent.png')
    plt.close(fig)
    plt.close(fig2)


    stk = pnl_data_concated.loc[feature_isfinite, :]
    signal_pnl = feature_data.loc[feature_isfinite]
    stk_index_label = pd.Index(stk.index.labels[0])
    stk_indicator = stk_index_label.duplicated(keep='first') ^ stk_index_label.duplicated(keep='last')  # sign for the beginning and the end of each ticker
    spread_bid_ask_array = params_dict['spread_bid_ask_array'][feature_isfinite]

    pnl_dict_nocost, pnl_dict_cost = dict(), dict()
    for u_l in params_dict['up_low_comb']:
        u, l = u_l
        pnl_dict_nocost['up_{}_low_{}'.format(u, l)], pnl_dict_cost['up_{}_low_{}'.format(u, l)], csv_res[' up_{}_low_{}_mean_holding_period'.format(u, l)] = af.back_test(signal_pnl, stk, stk_indicator, spread_bid_ask_array, u, l)
        gc.collect()
        del gc.garbage[:]
        gc.collect()
    pnl_nocost_concated = pd.concat(pnl_dict_nocost, axis=1, join='outer')
    pnl_cost_concated = pd.concat(pnl_dict_cost, axis=1, join='outer')
    pnc_unstack = pnl_nocost_concated.unstack(0)
    pc_unstack = pnl_cost_concated.unstack(0)
    pnls_average_no_cost = pnc_unstack.groupby(pnc_unstack.columns.get_level_values(0), axis=1).mean().mean(skipna=True, axis=1)
    pnl_average_cost = pc_unstack.groupby(pc_unstack.columns.get_level_values(0), axis=1).mean().mean(skipna=True, axis=1)
    pnls_min_nc_level = af.aggregate_to_min_return(pnls_average_no_cost, params_dict['time_level'])
    pnls_min_c_level = af.aggregate_to_min_return(pnl_average_cost, params_dict['time_level'])
    csv_res['sharpe_nc_daily'] = af.sharpe_min_level(pnls_min_nc_level.values, 10)
    csv_res['sharpe_c_daily'] =  af.sharpe_min_level(pnls_min_c_level.values, 10)
    csv_res['mdd_nc'] = af.mdd_ret_min_level(pnls_min_nc_level.values, 10)
    csv_res['mdd_c'] = af.mdd_ret_min_level(pnls_min_c_level.values, 10)
    feature_hd.close()
    target_hd.close()
    gc.collect()
    del gc.garbage[:]
    gc.collect()
    return pnls_average_no_cost, pnl_average_cost, csv_res, params_dict['feature_columns'][i]


def back_test_inside_functon_outside_without_pnl(i, params_dict):

    print('testing parameter:{}...'.format(params_dict['feature_columns'][i]))
    csv_res = dict()
    # fig, ax = plt.subplots(3, 3, figsize=(12, 9), dpi=120)
    # plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig2, ax2 = plt.subplots(3, 3, figsize=(12, 9), dpi=120)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    feature_hd = pd.HDFStore(params_dict['h5_result_name'], 'r')
    target_hd = pd.HDFStore(params_dict['h5_target_name'], 'r')
    feature_data = feature_hd[params_dict['feature_columns'][i]]
    feature_isfinite = np.isfinite(feature_data).values
    for j, target_name in enumerate(params_dict['target_scope']):
        target_raw = target_hd[target_name]
        target_isfinite = np.isfinite(target_raw).values
        intersct_vector = feature_isfinite & target_isfinite
        signal = feature_data.loc[intersct_vector]
        targt = target_raw.loc[intersct_vector]
        # corr_pearson = round(np.corrcoef(signal, targt)[0][1], 3)
        corr_spearman = round(sts.spearmanr(signal, targt)[0], 3)
        # csv_res[params_dict['target_scope'][j] + '_pearson_corr'] = corr_pearson
        csv_res[params_dict['target_scope'][j] + '_spearman_corr'] = corr_spearman
        # ret_sorted = af.return_sort_single(signal, targt)
        # ax[j // 3, j % 3].plot(np.arange(len(signal)), ret_sorted.cumsum())
        # ax[j // 3, j % 3].set_title(params_dict['target_scope'][j])
        return_percent = af.percentile_stats(signal, targt, 2, 2)
        ax2[j // 3, j % 3].bar(return_percent.index, return_percent.values.T[0])
        ax2[j // 3, j % 3].set_title(params_dict['target_scope'][j])
        del target_raw, target_isfinite, intersct_vector, signal, targt
        gc.collect()
        del gc.garbage[:]
        gc.collect()

    picure_name = params_dict['feature_columns'][i].replace(':', '_')
    # fig.suptitle(picure_name + '_correlation_plot')
    # fig.savefig(params_dict['output_rout'] + '/' + picure_name + '_correlation_plot.png')
    fig2.suptitle(picure_name + '_return_percent')
    fig2.savefig(params_dict['output_rout'] + '/' + picure_name + '_return_percent.png')
    # plt.close(fig)
    plt.close(fig2)
    feature_hd.close()
    target_hd.close()
    gc.collect()
    del gc.garbage[:]
    gc.collect()
    return csv_res, params_dict['feature_columns'][i]


# def backtest_pnl_info_part(i, params_dict):
#     print('pnl plotting:{}...'.format(params_dict['feature_columns'][i]))
#     csv_res = dict()
#     feature_hd = pd.HDFStore(params_dict['h5_result_name'], 'r')
#     target_hd = pd.HDFStore(params_dict['h5_target_name'], 'r')
#     feature_data = feature_hd[params_dict['feature_columns'][i]]
#     feature_isfinite = np.isfinite(feature_data).values
#     pnl_data_concated = target_hd['pnl_data_concated']

#     stk = pnl_data_concated.loc[feature_isfinite, :]
#     signal_pnl = feature_data.loc[feature_isfinite]
#     stk_index_label = pd.Index(stk.index.labels[0])
#     stk_indicator = stk_index_label.duplicated(keep='first') ^ stk_index_label.duplicated(keep='last')  # sign for the beginning and the end of each ticker
#     spread_bid_ask_array = params_dict['spread_bid_ask_array'][feature_isfinite]

#     pnl_dict_nocost, pnl_dict_cost = dict(), dict()

#     for u_l in params_dict['up_low_comb']:
#         u, l = u_l
#         pnl_dict_nocost['up_{}_low_{}'.format(u, l)], pnl_dict_cost['up_{}_low_{}'.format(u, l)], csv_res['mean_holding_time'], csv_res['trading_times'] = af.back_test(signal_pnl, stk, stk_indicator, spread_bid_ask_array, u, l)
#         gc.collect()
#         del gc.garbage[:]
#         gc.collect()
#     pnl_nocost_concated = pd.concat(pnl_dict_nocost, axis=1, join='outer')
#     pnl_cost_concated = pd.concat(pnl_dict_cost, axis=1, join='outer')
#     pnl_nocost_no_na = pnl_nocost_concated.dropna()
#     pnl_cost_no_na = pnl_cost_concated.dropna()
#     sum_ret_no_cost = np.sum(pnl_nocost_no_na.values)
#     sum_ret_cost = np.sum(pnl_cost_no_na.values)
#     csv_res['average_ret_per_trade_no_cost'] = sum_ret_no_cost / csv_res['trading_times']
#     csv_res['average_ret_per_trade_cost'] = sum_ret_cost / csv_res['trading_times']

#     picure_name = params_dict['feature_columns'][i].replace(':', '_')


#     pnl_hist_nocost = stats.trimboth(pnl_nocost_no_na.values[pnl_nocost_no_na.values != 0], 0.01)
#     pnl_hist_cost = stats.trimboth(pnl_cost_no_na.values[pnl_cost_no_na.values != 0], 0.01)


#     fig, ax = plt.subplots(1, 1, figsize=(9, 4))
#     ax.hist(pnl_hist_nocost, bins=700)
#     ax.set_title('Return Distribution No Cost')
#     fig.savefig(params_dict['output_rout'] + '/' + picure_name + '_return_distribtion_with_out_cost.png')
#     plt.close(fig)

#     fig, ax = plt.subplots(1, 1, figsize=(9, 4))
#     ax.hist(pnl_hist_cost, bins=700)
#     ax.set_title('Return Distribution With Cost')
#     fig.savefig(params_dict['output_rout'] + '/' + picure_name + '_return_distribtion_with_cost.png')
#     plt.close(fig)


#     pnc_unstack = pnl_nocost_concated.unstack(0)
#     pc_unstack = pnl_cost_concated.unstack(0)
#     pnls_average_no_cost = pnc_unstack.mean(skipna=True, axis=1)
#     pnl_average_cost = pc_unstack.mean(skipna=True, axis=1)
#     pnls_min_nc_level = af.aggregate_to_min_return(pnls_average_no_cost, params_dict['time_level'])
#     pnls_min_c_level = af.aggregate_to_min_return(pnl_average_cost, params_dict['time_level'])

#     feature_hd.close()
#     target_hd.close()
#     gc.collect()
#     del gc.garbage[:]
#     gc.collect()
#     return pnls_min_nc_level, pnls_min_c_level, csv_res, params_dict['feature_columns'][i]


def backtest_pnl_info_part(i, params_dict):
    print('pnl plotting:{}...'.format(params_dict['feature_columns'][i]))
    csv_res = dict()
    feature_hd = pd.HDFStore(params_dict['h5_result_name'], 'r')
    target_hd = pd.HDFStore(params_dict['h5_target_name'], 'r')
    feature_data = feature_hd[params_dict['feature_columns'][i]]
    feature_isfinite = np.isfinite(feature_data).values
    pnl_data_concated = target_hd['pnl_data_concated']

    stk = pnl_data_concated.loc[feature_isfinite, :]
    signal_pnl = feature_data.loc[feature_isfinite]
    stk_index_label = pd.Index(stk.index.labels[0])
    stk_indicator = stk_index_label.duplicated(keep='first') ^ stk_index_label.duplicated(keep='last')  # sign for the beginning and the end of each ticker
    spread_bid_ask_array = params_dict['spread_bid_ask_array'][feature_isfinite]

    pnl_dict_nocost, pnl_dict_cost = dict(), dict()

    for u_l in params_dict['up_low_comb']:
        u, l = u_l
        pnl_dict_nocost['up_{}_low_{}'.format(u, l)], pnl_dict_cost['up_{}_low_{}'.format(u, l)], csv_res = af.back_test(signal_pnl, stk, stk_indicator, spread_bid_ask_array, u, l)
        gc.collect()
        del gc.garbage[:]
        gc.collect()
    pnl_nocost_concated = pd.concat(pnl_dict_nocost, axis=1, join='outer')
    pnl_cost_concated = pd.concat(pnl_dict_cost, axis=1, join='outer')
    pnl_nocost_no_na = pnl_nocost_concated.dropna()
    pnl_cost_no_na = pnl_cost_concated.dropna()
    sum_ret_no_cost = np.sum(pnl_nocost_no_na.values)
    sum_ret_cost = np.sum(pnl_cost_no_na.values)

    picure_name = params_dict['feature_columns'][i].replace(':', '_')

    pnc_unstack = pnl_nocost_concated.unstack(0)
    pc_unstack = pnl_cost_concated.unstack(0)
    pnls_average_no_cost = pnc_unstack.mean(skipna=True, axis=1)
    pnl_average_cost = pc_unstack.mean(skipna=True, axis=1)
    pnls_min_nc_level = af.aggregate_to_min_return(pnls_average_no_cost, params_dict['time_level'])
    pnls_min_c_level = af.aggregate_to_min_return(pnl_average_cost, params_dict['time_level'])

    feature_hd.close()
    target_hd.close()
    gc.collect()
    del gc.garbage[:]
    gc.collect()
    return pnls_min_nc_level, pnls_min_c_level, csv_res, params_dict['feature_columns'][i]