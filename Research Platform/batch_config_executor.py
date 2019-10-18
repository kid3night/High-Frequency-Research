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
	
	
	# if 1:
	# 	args = parser.parse_args()
	# 	config_name = args.config_file
	# else:
	# 	config_name = 'Order_Direction_Amount'


	# config_name_list = ['Transaction_1_decay', 'Transaction_1_decay_Volume', 'Order_Direction_Volume_decay',
	# 					'Order_Direction_Amount_decay', 'tran_price_change_vol_decay', 'tran_price_change_points_decay', 
	# 					'Transaction_Cancellation_decay', 'Transaction_Cancellation_Square_decay',
	# 					'Transaction_Order_Percent_Diff', 'tran_type_num_diff']
	# config_name_list = ['Ask_Bid_Sum_Vol_decay', 'Transaction_Order_Percent_Diff', 'Ask_Bid_1_decay', 'Tran_Price_Change_Points',
	# 					'Order_Direction_Amount_decay', 'Transaction_Net_DIFF', 'Transaction_Order_Percent_Diff',
	# 					'Transaction_Order_Times_Diff']
	# config_name_list = ['Ask_Bid_MACD', 'Transaction_RSI_EMA', 'Transaction_KDJ', 'Transaction_CHO']
	# config_name_list = ['Transaction_VR', 'Transaction_ATR', 'Transaction_MASS', 'Transaction_UOS',]
	# config_name_list = ['Transaction_VR', 'Transaction_WAD',
	# 					'Transaction_DMI', 'Transaction_DMI_no_abs', 'Transaction_EMV',
	# 					'Transaction_CHO', 'Transaction_KDJ', 'Transaction_RSV', 'Transaction_RSI',
	# 					'Tran_Price_Change_Vol', 'Mid_Change_Origin', 'VRSI', 'RSI_TA', 'BIAS', 'PSY', 'Ask_Bid_CYR', 'Ask_Bid_OBV'
	# 					'Transaction_CYM', 'Ask_Bid_CYS', 'Ask_Bid_VMACD', 'Ask_Bid_1_decay', 'Ask_Bid_MACD', 'Tran_Type_Num_Diff', 
	#					'Transaction_Net_Vol', Order_Direction_Volume_decay]

	# config_name_list = ['Transaction_ACD',
	# 				  'Transaction_UOS',
	# 				  'Transaction_VR',
	# 				  'Transaction_WAD',
	# 				  'Transaction_DMI_no_abs',
	# 				  'Transaction_KDJ',
	# 				  'Transaction_RSV',
	# 				  'Tran_Price_Change_Vol',
	# 				  'Mid_Change_Origin',
	# 				  'VRSI',
	# 				  'RSI_TA',
	# 				  'BIAS',
	# 				  'Transaction_CYM',
	# 				  'Ask_Bid_CYS']
	# 				  #'Order_Average_Order',
	# 				  #'Transaction_Average_Order',
	# 				  #'Transaction_CHO'
	# 				  #'Transaction_RSI_EMA',
	# 				  #'Ask_Bid_1_New',
	# 				  #'Ask_Bid_Sum_Vol_decay',
	# 				  #'Order_Direction_Amount_decay',
	# 				  #'Transaction_Net_DIFF',
	# 				  #'Transaction_Order_Percent_Diff',
	# 				  #'Transaction_Order_Times_Diff',
	# 				  #'Transaction_EMV',
	# 				  #'Tran_Price_Change_Points',
	# 				  #'Ask_Bid_1_decay',
	# 				  #'Tran_Type_Num_Diff',
	# 				  #'Order_Direction_Volume_decay',
	# 				  #'Transaction_Net_Vol']


	#config_name_list = [
	#					'Ask_Bid_Sum_Vol_decay', 
	#					'Transaction_Net_DIFF', 'Transaction_EMV',
	#					'Ask_Bid_1_decay', 'Tran_Type_Num_Diff', 'Transaction_Net_Vol', 'Transaction_ACD', 'Transaction_UOS', 'Transaction_UDL', 'Transaction_VR', 
	#					'Transaction_Returns', 'Tran_Price_Change_Vol', 'Ask_Bid_1_New', 'Transaction_CYM', 'Ask_Bid_1_New2', 'Transaction_CHO']
	# config_name_list = ['Transaction_Order_Percent_Diff', 'Transaction_Order_Times_Diff', 'Order_Average_Order', 'Order_Direction_Amount_decay']
	# config_name_list = ['Transaction_CHO', 'Transaction_RSI_EMA', 'Transaction_AMV', 'Transaction_WVAD']
	# config_name_list = ['Transaction_ACD', 'Transaction_UOS', 'Transaction_OLD_UOS',
	# 					'Transaction_VR', 'Transaction_OLD_VR', 'Transaction_WAD',
	# 					'Transaction_KDJ', 'Transaction_RSV','Tran_Price_Change_Vol', 
	# 					'Mid_Change_Origin', 'VRSI', 'RSI_TA', 'BIAS', 'Transaction_CYM',
	# 					'Ask_Bid_CYS', 'Transaction_CHO', 'Ask_Bid_1_New', 'Ask_Bid_Sum_Vol_decay',
	# 					'Transaction_Net_DIFF', 'Transaction_EMV', 'Ask_Bid_1_decay', 'Tran_Type_Num_Diff',
	# 					'Transaction_UDL', 'Transaction_Returns', 'Transaction_DMI_no_abs', 'Transaction_AMV',
	# 					'Transaction_WVAD', 'Transaction_VPT', 'Ask_Bid_AMV', 'Transaction_Average_Order', 
	# 					'Order_Direction_Amount_decay', 'Order_Average_Order', 'Transaction_Order_Percent_Diff',
	# 					'Transaction_Order_Times_Diff']
	# config_name_list = ['Transaction_OLD_UOS', 'Transaction_OLD_VR', 'Transaction_VR', 'Transaction_UOS']
	# config_name_list = ['Order_Direction_Amount_decay', 
	# 					'Transaction_Order_Percent_Diff',
	# 					'Transaction_Order_Times_Diff',
	# 					'Order_Average_Order']
	# config_name_list = ['Transaction_Order_Times_Diff']
	# config_name_list = [] 
	# config_name_list = ['VRSI', 'Ask_Bid_1_New']
	# config_name_list = ['VRSI']
	# config_name_list = ['Ask_Bid_1_New']
	# config_name_list = ['Mid_Change_Origin']
	config_name_list = ['Transaction_Returns']

	for config_name in config_name_list:	

		config_path = './config_select_update/' + config_name + '.json'

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




