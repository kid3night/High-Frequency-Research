#log_transformer.py
import pandas as pd
import numpy as np
import h5py


def get_keysHDF(path):

    f = h5py.File(path)
    keys = list(f.keys())
    return keys

f_vol_type = lambda x: np.nan if np.isnan(x) else 0 if x == 0 else np.log(x) if x > 0 else -np.log(-(x))
f_rate_type = lambda x: np.nan if np.isnan(x) else 0 if x == 0 else np.log(x + 1) if x > 0 else -np.log(-(x - 1))
f_vol_type_V = np.vectorize(f_vol_type)
f_rate_type_V = np.vectorize(f_rate_type)


def transform_feature(feature_path, transform_sign):

	feature_path_split = feature_path.split('/')
	feature_path_split[-1] = 'log_transformed_' + feature_path_split[-1]
	output_path = "/".join(feature_path_split)
	data_hd = pd.HDFStore(feature_path, 'r')
	d_keys = get_keysHDF(feature_path)
	for key in d_keys:
		if 'nperiod' in key:
			print(key)
			data = data_hd[key]
			if transform_sign == 'vol':
				result_series = pd.Series(f_vol_type_V(data.values), index=data.index)
				result_series.to_hdf(output_path, mode='a', key=key)
			elif transform_sign == 'rate':
				result_series = pd.Series(f_vol_type_V(data.values), index=data.index)
				result_series.to_hdf(output_path, mode='a', key=key)
			elif transform_sign == 'negative':
				result_series = -data
				result_series.to_hdf(output_path, mode='a', key=key)
			else:
				print('transform_sign incorrect!!')
		else:
			result_series = data_hd[key]
			result_series.to_hdf(output_path, mode='a', key=key)




if __name__ == '__main__':
	feature_path = 'F:/TOPT_0919/Test_Order_Direction_Amount_decay_On_ZZ500_From_20180601_To_20180801/Order_Direction_Amount_decay.h5'
	transform_sign = 'negative'
	transform_feature(feature_path, transform_sign)