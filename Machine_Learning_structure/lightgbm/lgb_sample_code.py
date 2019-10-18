import pandas as pd
import numpy as np
import lightgbm as lgb
from machine_learning_class import machine_learning_class
import machine_learning_class as ml
from datetime import datetime
import scipy.stats as sts
import matplotlib.pyplot as plt


if __name__ == '__main__':

	###  model train
	train_data_path = 'data_sample.h5'
	target_name = 'Target'
	ml_obj = machine_learning_class(train_data_path, target_name, select_key='X')
	ml_obj.get_X_Y_train()
	data_train = lgb.Dataset(ml_obj.X_train, ml_obj.y_train, silent=True)
	data_eval = lgb.Dataset(ml_obj.X_test, ml_obj.y_test, reference=data_train)

	### setting model's training parameters important! 
	model_1_params = {'application': 'regression', 
	                    'boosting':'gbdt',
	                    'num_iterations':40,
	                    'learning_rate':0.05,
	                    'max_depth':10,
	                    'num_leaves':40,
	                    'verbose':2, 
	                    'feature_fraction':0.7,
	                    'bagging_fraction':0.7,
	                    'bagging_freq':5,
	                    'min_data_in_leaf':10,
	                    'lambda_l2':1,
	                    'num_threads':15,
	                    'early_stopping_round':10,
	                    'metric':'l2'}

	start = datetime.now()
	# lgb_model = lgb.train(model_1_params, data_train, valid_sets=data_eval)

########
	def corr_metric(y_hat, data):
	    y_real = data.get_label()
	    corr = np.corrcoef(y_hat, y_real)[0][1]
	    return 'Correlation', corr, True

	lgb_model = lgb.train(model_1_params, data_train, valid_sets=[data_eval, data_train], feval=corr_metric, 
	                      valid_names=['val', 'train'], learning_rates=lambda iter: 0.05 * (0.999 ** iter),
	                      evals_result = {})
 # The code in the section is to show how to define a metric for evaluate the training result during the model training process
 # learning_rates=lambda iter: 0.05 * (0.999 ** iter) is used to control the learning rate decay. With the growing of iteration nums,
 # the learning rate can be reduced.
########


	end = datetime.now()
	total_minute = (end - start).days * 24 * 60 + (end - start).seconds / 60
	print('total training time is {} minutes'.format(total_minute))
	lgb_model.save_model('sample_model' + '.txt')


	###  model predict
	print('use the model to predict...')
	data_ = pd.read_hdf('data_sample.h5', key='X')
	data_input = data_.loc[:, data_.columns != 'Target']
	data_input_target = data_.loc[:, data_.columns == 'Target'].values
	lgb_model = lgb.Booster(model_file='sample_model.txt') 
	y_testset_predict = lgb_model.predict(data_input.values, num_iteration=lgb_model.best_iteration)
	# plt.plot(y_testset_predict, data_input_target, '.')
	# plt.show()

