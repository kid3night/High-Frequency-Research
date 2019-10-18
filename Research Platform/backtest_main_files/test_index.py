import pandas as pd
import numpy as np
import backtest_engine_initial_script as bei
import get_universe as gu
import time

if __name__ == '__main__':

	index_path = 'F:/TANG_NEW/000905weightnextday20180816.xls'
	universe, diff = gu.verify_tickers(index_path, 0)
	print('tickers of which data is not included: {}'.format(diff))	
	universe_name = 'test_500_consititute_2_month'

	# universe = ["000001.SZ","000002.SZ","000004.SZ","000005.SZ","000006.SZ"]
	# universe_name = 'test_on_small_set'
	start_time = 20180601
	end_time = 20180801
	output_path = 'F:/Test_Out_Put'
	feature_name = 'Transaction_1'
	multi_proc_feature = 10
	multi_proc_result = 5

	no_pnl = 1
	feature_file_name = 'sample_feature'
	params = {'nperiod':[3, 5, 8, 10, 13]}
	feature_type = ['transaction']
	each_type_data_need = {'transaction':['Turnover', 'BSFlag']}
	# saved_data_path = 'F:/Test_Out_Put/Test_Ask_Bid_1_On_test_500_consititute_From_20180601_To_20180701/'
	# saved_data_path = 'F:/Test_Out_Put/Test_Ask_Bid_1_On_test_500_consititute_3_month_From_20180601_To_20180801/'
	saved_data_path = None
	new_result_folder_name = 'result_no_pnl'
	operation_type = "full_backtest"

	bei.backtesting_main(start_time, end_time, output_path, feature_name, multi_proc_feature, 
	                	 multi_proc_result, no_pnl, feature_file_name, feature_type, each_type_data_need,
	                	 params, universe, universe_name, new_result_folder_name, operation_type, saved_data_path)
