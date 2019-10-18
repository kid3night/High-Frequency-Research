from backtest_main_files.base_transform import base_transform
import pandas as pd
import numpy as np
import talib as ta


class Transform_EMA(base_transform):

	def transform_function(self, data):

		def EMA_15(x):
			x = x.ffill()
			result_array = ta.EMA(x.values, 15)
			return pd.Series(result_array, index=x.index)

		transformed_feature = data.groupby(level=0).apply(EMA_15)
		return transformed_feature



class Transform_Ask_Bid_1_New(base_transform):

	def transform_function(self, data):

		transformed_feature = (data * 10) ** 2
		return transformed_feature



if __name__ == '__main__':

	feature_dir = 'F:/TOPT_1009/Test_Ask_Bid_1_New_On_ZZ500_From_20180601_To_20180801'
	feature_name = 'Ask_Bid_1_New'
	feature_new_subfolder = 'multiply_square'
	feature_transform_obj = Transform_Ask_Bid_1_New(feature_dir, feature_name, feature_new_subfolder)
	feature_transform_obj.feature_transform()
