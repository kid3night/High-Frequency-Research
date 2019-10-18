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


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, required=True, help='name of the configuration file to be tested')

if __name__ == '__main__':
	
	if 1:
		args = parser.parse_args()
		config_name = args.config_file
	else:
		config_name = 'Order_Direction_Amount'
	config_path = './config_ML/' + config_name + '.json'

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



