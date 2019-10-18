# import sys
# sys.path.append('../')
# print(sys.path)
import json
import pandas as pd
import numpy as np
import backtest_main_files.backtest_engine_initial_script as bei
import backtest_main_files.get_universe as gu
import time
from pprint import pprint
import argparse
import gc


if __name__ == '__main__':
	
	
	# config_name_list = [
	# 					# 'Ask_Bid_1_New','Ask_Bid_AMV',
	# 					# 'Ask_Bid_CYS','Ask_Bid_New_update',
	# 					# 'Ask_Bid_Sum_Vol_decay','BIAS',
	# 					# 'Mid_Change_Origin','Order_Average_Order','Order_Direction_Amount_decay',
	# 					# 'RSI_TA','Transaction_ACD','Transaction_AMV','Transaction_Average_Order',
	# 					# 'Transaction_CHO','Transaction_CYM','Transaction_DMI_no_abs','Transaction_EMV',
	# 					# 'Transaction_KDJ','Transaction_Net_DIFF',

	# 					# 'Transaction_Net_Vol',
	# 					# 'Transaction_Order_Percent_Diff','Transaction_Order_Times_Diff',
	# 					# 'Transaction_Returns','Transaction_RSV','Transaction_UDL',
	# 					# 'Transaction_UOS','Transaction_VPT','Transaction_VR','Transaction_WAD',
	# 					# 'Transaction_WVAD','Tran_Price_Change_Vol','Tran_Type_Num_Diff','VRSI','VRSI_update'
	# 					'Ask_Bid_1_decay']

	# config_name_list = ['Transaction_VPT','Transaction_VR','Transaction_WAD',
	# 					'Transaction_WVAD','Tran_Price_Change_Vol',
	# 					'Tran_Type_Num_Diff','VRSI','VRSI_update']
	# config_name_list = ['Ask_Bid_1_decay']
	# config_name_list = ['Transaction_OLD_VR', 'Transaction_OLD_UOS']
	# config_name_list = [
	# 					# 'Transaction_alpha101_22',
	# 					# 'Transaction_alpha101_25',
	# 					# 'Transaction_alpha101_32',
	# 					# 'Transaction_alpha101_53',
	# 					# 'Transaction_alpha101_55',
	# 					'Transaction_alpha101_61'
	# 					# 'Transaction_alpha101_84',
	# 					# 'Transaction_Corr_Adjusted_Returns',
	# 					# 'Transaction_delta_VOL_Adjusted_Returns',
	# 					# 'Transaction_LON',
	# 					# 'Transaction_OLD_UOS',
	# 					# 'Transaction_OLD_VR',
	# 					# 'Transaction_price_skewness',
	# 					# 'Transaction_vol_skewness',
	# 					# 'Transaction_vol_skewness_corr',
	# 					# 'Transaction_ZLJC',
	# 					# 'Transaction_alpha101_22'
	# 					]
	config_name_list = [
						# 'Transaction_alpha101_25',
						# 'Transaction_alpha101_32',
						'Transaction_alpha101_61',
						'Transaction_Corr_Adjusted_Returns',
						'Transaction_delta_VOL_Adjusted_Returns',
						'Transaction_price_skewness',
						'Transaction_ZLJC',
						]

	for config_name in config_name_list:	

		config_path = './config_new_feature_validate_1203/' + config_name + '.json'
		# config_path = './config_1203_Test/' + config_name + '.json'

		data_hd = open(config_path).read()
		config_data = json.loads(data_hd)

		for key in config_data.keys():
			data_ = str(config_data[key])
			exec(key + '=' + data_)

		if import_universe:
			if 'order' in feature_type or 'orderbook' in feature_type:
				universe, diff = gu.verify_tickers(universe_file_name, 1)
			else:
				universe, diff = gu.verify_tickers(universe_file_name, 0)
			print('tickers of which data is not included: {}'.format(diff))	
		else:
			universe = self_specified_universe
			if 'order' in feature_type or 'orderbook' in feature_type:
				universe, diff = gu.get_intersect_diff(universe, 1)
			else:
				universe, diff = gu.get_intersect_diff(universe, 0)
			print('tickers of which data is not included: {}'.format(diff))	


		bei.backtesting_main(test_time_start, test_time_end, test_result_save_path, feature_name, multi_process_feature, 
							 multi_process_result, not_plot_pnl, feature_file_name, feature_type, each_type_data_need,
							 feature_params, universe, universe_name, new_result_subfolder_name, operation_type, 
							 generated_feature_h5_path, generated_target_h5_path)
