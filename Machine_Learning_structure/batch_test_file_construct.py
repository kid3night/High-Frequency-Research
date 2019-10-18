import pandas as pd
import numpy as np


train_data_path = 'F:/Machine_Learning_Structure/LightGBM_Tuning/SZ_m6_m8.h5'
test_data_path = 'F:/Machine_Learning_Structure/LightGBM_Tuning/SZ_m8_m9.h5'

data_predict_train = pd.read_hdf('lgb_predictions.h5'， key='train')
data_predict_test = pd.read_hdf('lgb_predictions.h5'， key='test')

data_raw_train = pd.read_hdf(train_data_path, 'X')
data_raw_test = pd.read_hdf(test_data_path, 'X')

new_data_train = np.zeros((data_raw_train.shape[0], data_predict_train.shape[1]))
new_data_test = np.zeros((data_raw_test.shape[0], data_predict_test.shape[1]))
new_data_train[:] = np.nan
new_data_test[:] = np.nan
train_raw_finite_pos = np.isfinite(data_raw_train).all(axis=1)
test_raw_finite_pos = np.isfinite(data_raw_test).all(axis=1)
new_data_train[train_raw_finite_pos, :] = data_predict_train.values
new_data_test[test_raw_finite_pos, :] = data_predict_test.values

spread_6_8_hd = pd.HDFStore('F:/TOPT_1029/Test_Ask_Bid_Spread_rate_On_ZZ5002_1_From_20180601_To_20180801/Ask_Bid_Spread_rate.h5', 'r')
spread_6_8 =spread_6_8_hd['nperiod:10'].values.reshape(-1, 1)
spread_8_9_hd = pd.HDFStore('F:/TOPT_08_09_Validate/Test_Ask_Bid_Spread_rate_On_ZZ5002_1_From_20180802_To_20180903/Ask_Bid_Spread_rate.h5', 'r')
spread_8_9 =spread_8_9_hd['nperiod:10'].values.reshape(-1, 1)

name_list = ['nperiod:{}'.format(i) for i in range(2, 9)]
assert new_data_train.shape[0] == spread_6_8.shape[0] and spread_6_8.shape[1] == 1 and new_data_train.shape[1] == len(name_list)
temp_minus = np.abs(new_data_train) - spread_6_8
new_res_train = np.sign(new_data_train) * np.where(temp_minus > 0, temp_minus, 0)
df_train = pd.DataFrame(new_res_train, columns=name_list, index=data_raw_train.index)

assert new_data_test.shape[0] == spread_8_9.shape[0] and spread_8_9.shape[1] == 1 and new_data_test.shape[1] == len(name_list)
temp_minus = np.abs(new_data_test) - spread_8_9
new_res_test = np.sign(new_data_test) * np.where(temp_minus > 0, temp_minus, 0)
df_test = pd.DataFrame(new_res_test, columns=name_list, index=data_raw_test.index)

for name in name_list:
	df_train[name].to_hdf('trained_predicts.h5', key=name)
	df_test[name].to_hdf('tested_predicts.h5', key=name)

spread_6_8_hd['other_paras'].to_hdf('trained_predicts.h5', key='other_paras')
spread_8_9_hd['other_paras'].to_hdf('tested_predicts.h5', key='other_paras')