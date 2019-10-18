import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import scipy.stats as sts

class machine_learning_class:

	def __init__(self, feature_path, target_name, select_key='X', n_Kfold=3, split_random_seed=10, test_size=0.2):

		self.feature_path = feature_path
		self.target_name = target_name
		self.n_Kfold = n_Kfold
		self.test_size_ = test_size
		self.split_random_seed = split_random_seed
		self.select_key = select_key

	def get_X_Y_train(self):

		feature_hd = pd.HDFStore(self.feature_path, 'r')
		selected_part = feature_hd[self.select_key]
		feature_hd.close()
		X = selected_part.loc[:, selected_part.columns != self.target_name].values
		y = selected_part.loc[:, selected_part.columns == self.target_name].values.flatten()
		del selected_part
		gc.collect()
		del gc.garbage[:]
		gc.collect()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size_, random_state=self.split_random_seed)
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		print('X_train:shape {}'.format(self.X_train.shape))
		print('y_train:shape {}'.format(self.y_train.shape))
		print('X_test:shape {}'.format(self.X_test.shape))
		print('y_test:shape {}'.format(self.y_test.shape))


	def get_X_Y_test(self, test_data_path):

		feature_hd = pd.HDFStore(test_data_path, 'r')
		selected_part = feature_hd['X']
		feature_hd.close()
		X = selected_part.loc[:, selected_part.columns != self.target_name].values
		y = selected_part.loc[:, selected_part.columns == self.target_name].values.flatten()
		del selected_part
		gc.collect()
		del gc.garbage[:]
		gc.collect()
		return (X, y)


	def train_k_fold(self):

		kf = KFold(n_splits = self.n_Kfold, random_state=None, shuffle=True)
		score_array_train = []
		score_array_test = []
		for train_index, test_index in kf.split(X_train):
			X_train_run, X_validate = self.X_train[train_index], self.X_train[test_index]
			y_train_run, y_validate = self.y_train[train_index], self.y_train[test_index]
			ml_model.fit(X_train_run, y_train_run)
			score_array_train_run.append(ml_model.score(X_train_run, y_train_run))
			score_array_validate.append(ml_model.score(X_validate, y_validate))
		score_train_mean = np.mean(score_array_train)
		score_test_mean = np.mean(score_array_test)
		return score_overall


	def random_search(self, random_grid, param_combo_number):

		self.model_random = RandomizedSearchCV(estimator = self.ml_model, 
									      param_distributions = random_grid,
									      n_iter = param_combo_number, cv = self.n_Kfold, 
									      verbose=2, n_jobs = -1)
		self.model_random.fit(self.X_train, self.y_train)
		print(self.model_random.best_params_)


def model_train_lgb(params, X_train, y_train, model_name, save_path='.'):
	
	data_train = lgb.Dataset(X_train, y_train, silent=True)
	start = datetime.now()
	lgb_model = lgb.train(params, data_train, params['num_iterations'])
	end = datetime.now()
	total_minute = (end - start).days * 24 * 60 + (end - start).seconds / 60
	print('total training time is {} minutes'.format(total_minute))
	lgb_model.save_model(save_path + '/' + model_name + '.txt')
	return lgb_model


def model_test_lab(model_lgb, X, y):

	y_predict = model_lgb.predict(X)
	pearson_corr = np.corrcoef(y_predict, y.flatten())[0, 1]
	spearman_corr = sts.spearmanr(y_predict, y.flatten())[0]
	print('Pearson Correlation is {}, Spearman Correlation is {}'.format(pearson_corr, spearman_corr))
	return y_predict
