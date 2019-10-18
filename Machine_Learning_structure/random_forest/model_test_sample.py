import pandas as pd
import numpy as np
import machine_learning_class as mlc
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# setting training parameters
parameters = {'min_samples_split': 5, 'verbose':2, 'n_jobs': 1, 
              'max_features': 'auto', 'n_estimators':10, 'max_depth':10}
feature_file_path = 'data_sample.h5'
target_name = 'Target'
# generate model object
ml_model = RandomForestRegressor(**parameters)
test_obj = mlc.machine_learning_class(feature_file_path, target_name, ml_model, test_size=0.3)
test_obj.get_X_Y()
# train the model
test_obj.ml_model.fit(test_obj.X_train, test_obj.y_train)
joblib.dump(test_obj.ml_model, 'test_model.m')
# load model
model_generated = joblib.load('test_model.m')
print('use the model to predict...')
data_ = pd.read_hdf('data_sample.h5', key='X')
data_input = data_.loc[:, data_.columns != 'Target']
data_input_target = data_.loc[:, data_.columns == 'Target'].values
result_predict = model_generated.predict(data_input.values)
# plt.plot(result_predict, data_input_target, '.')
# plt.show()