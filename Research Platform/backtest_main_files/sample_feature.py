# import sys
# sys.path.append('./backtest_main_files/')
# print(sys.path)
import pandas as pd
import numpy as np
from baseSignal import base_feature

class Ask_Bid_1(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['tick_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				return (data[-1] - data[0]) / len(data)

		bid_part = ['BidVol10', 'BidVol9', 'BidVol8', 'BidVol7', 'BidVol6',
					'BidVol5', 'BidVol4', 'BidVol3', 'BidVol2', 'BidVol1']
		ask_part = ['AskVol1', 'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5',
					'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10']
		rolling_time = str(nperiod * 3) + 'S'
		vol_diff = stk_data.loc[:, bid_part].sum(axis=1) - stk_data.loc[:, ask_part].sum(axis=1)
		return vol_diff.rolling(rolling_time, closed='right').apply(function_inside, raw=True)



class Transaction_1(base_feature):
	param_list = ['nperiod']

	def feature(self, data_fed, nperiod):

		stk_data = data_fed['transaction_data']

		def function_inside(data):
			if len(data) == 0:
				return 0
			else:
				return data.sum()

		rolling_time = str(nperiod * 3) + 'S'
		bs_multiplier = np.where(stk_data['BSFlag'].values == 66, 1, np.where(stk_data['BSFlag'].values == 83, -1, 0))
		signed_turnover = stk_data['Turnover'] * bs_multiplier
		return signed_turnover.rolling(rolling_time,  closed='left').apply(function_inside, raw=True)


