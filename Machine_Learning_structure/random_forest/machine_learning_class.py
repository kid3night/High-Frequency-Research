import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


class machine_learning_class:

	def __init__(self, feature_path, target_name, ml_model, n_Kfold=3, split_random_seed=10, test_size=0.3):

		self.feature_path = feature_path
		self.target_name = target_name
		self.n_Kfold = n_Kfold
		self.ml_model = ml_model
		self.test_size_ = test_size
		self.split_random_seed = split_random_seed

	def get_X_Y(self):
		feature_hd = pd.HDFStore(self.feature_path, 'r')
		selected_part = feature_hd['X']
		feature_hd.close()
		X = pd.DataFrame(selected_part.loc[:, selected_part.columns != self.target_name].values)
		y = pd.Series(selected_part.loc[:, selected_part.columns == self.target_name].values.flatten())
		del selected_part
		gc.collect()
		del gc.garbage[:]
		gc.collect()
		X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(X.index, y.index, test_size=self.test_size_, random_state=self.split_random_seed)
		X_train = X.loc[X_train_index]
		X_test = X.loc[X_test_index]
		y_train = y.loc[y_train_index]
		y_test = y.loc[y_test_index]

		self.X_train = X_train.values
		self.X_test = X_test.values
		self.y_train = y_train.values.flatten()
		self.y_test = y_test.values.flatten()
		print('X_train:shape {}'.format(self.X_train.shape))
		print('y_train:shape {}'.format(self.y_train.shape))
		print('X_test:shape {}'.format(self.X_test.shape))
		print('y_test:shape {}'.format(self.y_test.shape))



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



# if __name__ == '__main__':
	

# 	n_estimators = list(range(100, 3100, 100)) 
# 	max_features = ['auto', 'sqrt']
# 	max_depth = [int(x) for x in np.linspace(50, 550, num = 50)]
# 	max_depth.append(None)
# 	min_samples_split = [1000, 5000, 10000, 20000, 50000, 100000]

# 	random_grid = {'n_estimators': n_estimators,
# 	               'max_features': max_features,
# 	               'max_depth': max_depth,
# 	               'min_samples_split': min_samples_split,
# 	               'min_samples_leaf': min_samples_leaf}

# 	param_combo_number = 100

# 	ml_model = RandomForestRegressor()
# 	test_obj = machine_learning_cv('SZ', 'MidReturn60', 3, ml_model)
# 	test_obj.get_X_Y()
# 	test_obj.random_search(random_grid, param_combo_number)




