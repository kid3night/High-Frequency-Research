# import sys
# sys.path.append('./backtest_main_files/')
# print(sys.path)
import pandas as pd
import numpy as np
import backtest_engine_final as bef
import time
import gc


def backtesting_data_generate_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                                   multi_proc_result, no_pnl, feature_file_name, feature_type,
                                   each_type_data_need, params, universe, universe_name):
    bte = bef.backtest_engine(start_time=start_time, end_time=end_time, output_path=output_path, 
                              feature_name=feature_name, multi_proc_feature=multi_proc_feature, 
                              multi_proc_result=multi_proc_result, no_pnl=no_pnl, feature_type=feature_type,
                              each_type_data_need=each_type_data_need)
    bte.get_feature_obj(feature_file_name=feature_file_name, params=params)
    bte.feature_target_pair_generator(universe)
    bte.create_rout(universe_name)
    default_feature_path = bte.output_rout
    return default_feature_path


def backtesting_backtest_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                              multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need, 
                              params,universe, universe_name, new_result_folder_name, saved_feature_path, saved_target_path):
    
    bte = bef.backtest_engine(start_time=start_time, end_time=end_time, output_path=output_path, 
                          feature_name=feature_name, multi_proc_feature=multi_proc_feature, feature_type=feature_type,
                          each_type_data_need=each_type_data_need, multi_proc_result=multi_proc_result, no_pnl=no_pnl)

    bte.read_from_data_backtest(feature_file_name=feature_file_name, params=params, 
                                saved_feature_path=saved_feature_path, saved_target_path=saved_target_path,
                                new_result_folder_name=new_result_folder_name, universe_name=universe_name)

    
def backtesting_main(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                     multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need, params,
                     universe, universe_name, new_result_folder_name, operation_type, saved_feature_path=None, saved_target_path=None):

    if operation_type == "feature_calculation":
        start = time.clock()
        print('======under feature calculation only mode...======')
        feature_data_path = backtesting_data_generate_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                                                           multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need,
                                                           params, universe, universe_name)
        print('feature_calculation used {}s'.format(round(time.clock() - start, 2)))
    elif operation_type == "full_backtest":
        print('======under full backtest mode...======')
        start = time.clock()
        feature_data_path = backtesting_data_generate_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                                                           multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need, 
                                                           params, universe, universe_name)
        gc.collect()
        del gc.garbage[:]
        gc.collect()
        print('feature_calculation used {}s'.format(round(time.clock() - start, 2)))
        start = time.clock()
        print(new_result_folder_name)
        backtesting_backtest_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                                  multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need, params,
                                  universe, universe_name, new_result_folder_name, saved_feature_path=feature_data_path,
                                  saved_target_path=feature_data_path)
        print('backtesting used {}s'.format(round(time.clock() - start, 2)))
    else:
        print('======under backtest only mode...======')
        start = time.clock()
        assert saved_feature_path is not None and saved_target_path is not None
        backtesting_backtest_part(start_time, end_time, output_path, feature_name, multi_proc_feature, 
                                  multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need, 
                                  params, universe, universe_name, new_result_folder_name, saved_feature_path, saved_target_path)
        print('backtesting used {}s'.format(round(time.clock() - start, 2)))