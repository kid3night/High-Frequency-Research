from abc import ABCMeta, abstractmethod
import itertools as it
import warnings
# warnings.simplefilter(action='ignore', category='NaturalNameWarning')
import pandas as pd
import os
import h5py


def get_keysHDF(path):

	f = h5py.File(path)
	keys = list(f.keys())
	return keys


class base_transform:

	__metaclass__ = ABCMeta

	def __init__(self, feature_dir, feature_name, feature_new_subfolder):

		self.feature_dir = feature_dir
		self.feature_name = feature_name
		self.feature_path = feature_dir + '/' + feature_name + '.h5'
		self.feature_new_dir = feature_dir + '/' + feature_new_subfolder
		if not os.path.exists(self.feature_new_dir):
		    os.makedirs(self.feature_new_dir)
		self.feature_new_path = self.feature_new_dir + '/' + feature_name + '.h5'

	def feature_transform(self):

		data_hd = pd.HDFStore(self.feature_path, 'r')
		d_keys = get_keysHDF(self.feature_path)
		for key in d_keys:
			print('processing key:{}'.format(key))
			data = data_hd[key]
			if 'nperiod' in key:
				result_series = self.transform_function(data)
				result_series.to_hdf(self.feature_new_path, mode='a', key=key)
			else:
				data.to_hdf(self.feature_new_path, mode='a', key=key)
		data_hd.close()

	@abstractmethod
	def transform_function(self, data):
		"""
		This is the thing to override. Research should define their logic of this transform function
		"""
		pass










