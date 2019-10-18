import pandas as pd
import numpy as np
import lightgbm as lgb

data_hd = pd.HDFStore('feature_matrix_test_set.h5', 'r')
target_name = 'Target_60_Seconds_MidReturn60'
data_feature = data_hd['clean_feature']
X = data_feature.loc[:, data_feature.columns != target_name].values

lgb_model1 = lgb.Booster(model_file='lgbmodel1.txt') 
lgb_model2 = lgb.Booster(model_file='lgbmodel2.txt') 
lgb_model3 = lgb.Booster(model_file='lgbmodel3.txt') 

y_predict1 = lgb_model1.predict(X)
y_predict2 = lgb_model2.predict(X)
y_predict3 = lgb_model3.predict(X)

y_all = np.concatenate((y_predict1.reshape(-1, 1), y_predict2.reshape(-1, 1), y_predict3.reshape(-1, 1)), axis=1)
spread_data = pd.read_hdf('F:/TOPT_08_09_Validate/Test_Ask_Bid_Spread_rate_On_ZZ5002_1_From_20180802_To_20180903/Ask_Bid_Spread_rate.h5', 'nperiod:10')
pos_finite = pd.read_hdf('feature_matrix_test_set.h5', 'finite_pos')
spread_data_SZ = spread_data.iloc[:25300981]

new_data_test = np.zeros((spread_data_SZ.shape[0], y_all.shape[1]))
new_data_test[:] = np.nan
new_data_test[pos_finite.values, :] = y_all

name_list = ['nperiod:{}'.format(i) for i in range(1, 4)]
features = pd.DataFrame(new_data_test, index=spread_data_SZ.index, columns=name_list)
other_paras_data = pd.read_hdf('F:/TOPT_08_09_Validate/Test_Ask_Bid_Spread_rate_On_ZZ5002_1_From_20180802_To_20180903/Ask_Bid_Spread_rate.h5', 'other_paras')

other_paras_data[:] = 25300981

name_list = ['nperiod:{}'.format(i) for i in range(1, 4)]
for name in name_list:
    feature_d = features[name]
    sign_feature_d = np.sign(feature_d.values)
    abs_minus_spread = np.abs(feature_d.values) - spread_data_SZ.values
    result = np.where(abs_minus_spread > 0, abs_minus_spread, 0) * sign_feature_d
    result_series = pd.Series(result, index=feature_d.index)
    result_series.to_hdf('./backtest/Testset/SZ_feature.h5', key=name)
other_paras_data.to_hdf('./backtest/Testset/SZ_feature.h5', key='other_paras')

target_data = pd.HDFStore('F:/TOPT_08_09_Validate/Test_Ask_Bid_Spread_rate_On_ZZ5002_1_From_20180802_To_20180903/Ask_Bid_Spread_rate_target.h5', 'r')

for i in target_data.keys():
    data_ = target_data[i].iloc[:25300981]
    data_.to_hdf('./backtest/Testset/SZ_feature_target.h5', key=i)