import pandas as pd
import numpy as np
import gc

path_dict_1 = {'F:/TOPT_08_09_Validate/Test_Transaction_ACD_On_ZZ500__From_20180802_To_20180903':'Transaction_ACD', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_OLD_UOS_On_ZZ500__From_20180802_To_20180903':'Transaction_OLD_UOS', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_UOS_On_ZZ500__From_20180802_To_20180903':'Transaction_UOS', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_VR_On_ZZ500__From_20180802_To_20180903':'Transaction_VR', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_OLD_VR_On_ZZ500__From_20180802_To_20180903':'Transaction_OLD_VR', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_WAD_On_ZZ500_From_20180802_To_20180903':'Transaction_WAD', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_KDJ_On_ZZ500__From_20180802_To_20180903':'Transaction_KDJ', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_RSV_On_ZZ500__From_20180802_To_20180903':'Transaction_RSV', 
			   'F:/TOPT_08_09_Validate/Test_Tran_Price_Change_Vol_On_ZZ500__From_20180802_To_20180903':'Tran_Price_Change_Vol', 
			   'F:/TOPT_08_09_Validate/Test_Mid_Change_Origin_On_ZZ500__From_20180802_To_20180903':'Mid_Change_Origin', 
			   'F:/TOPT_08_09_Validate/Test_VRSI_On_ZZ500__From_20180802_To_20180903':'VRSI', 
			   'F:/TOPT_08_09_Validate/Test_RSI_TA_On_ZZ500__From_20180802_To_20180903':'RSI_TA', 
			   'F:/TOPT_08_09_Validate/Test_BIAS_On_ZZ500__From_20180802_To_20180903':'BIAS', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_CYM_On_ZZ500__From_20180802_To_20180903':'Transaction_CYM', 
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_CYS_On_ZZ500__From_20180802_To_20180903':'Ask_Bid_CYS', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_CHO_On_ZZ500__From_20180802_To_20180903':'Transaction_CHO', 
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_1_New_On_ZZ500__From_20180802_To_20180903':'Ask_Bid_1_New', 
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_Sum_Vol_decay_On_ZZ500__From_20180802_To_20180903':'Ask_Bid_Sum_Vol_decay', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_Net_DIFF_On_ZZ500__From_20180802_To_20180903':'Transaction_Net_DIFF', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_EMV_On_ZZ500__From_20180802_To_20180903':'Transaction_EMV', 
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_1_decay_On_ZZ500__From_20180802_To_20180903':'Ask_Bid_1_decay', 
			   'F:/TOPT_08_09_Validate/Test_Tran_Type_Num_Diff_On_ZZ500__From_20180802_To_20180903':'Tran_Type_Num_Diff', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_UDL_On_ZZ500__From_20180802_To_20180903':'Transaction_UDL', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_Returns_On_ZZ500__From_20180802_To_20180903':'Transaction_Returns', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_DMI_no_abs_On_ZZ5002_From_20180802_To_20180903':'Transaction_DMI_no_abs', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_AMV_On_ZZ500__From_20180802_To_20180903':'Transaction_AMV', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_WVAD_On_ZZ500__From_20180802_To_20180903':'Transaction_WVAD', 
			   'F:/TOPT_08_09_Validate/Test_Transaction_VPT_On_ZZ500__From_20180802_To_20180903':'Transaction_VPT', 
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_AMV_On_ZZ500__From_20180802_To_20180903':'Ask_Bid_AMV',
			   'F:/TOPT_08_09_Validate/Test_Transaction_UOS_On_ZZ500__From_20180802_To_20180903/target':'Target_60_Seconds',
			   'F:/TOPT_08_09_Validate/Test_Transaction_Average_Order_On_ZZ500__From_20180802_To_20180903':'Transaction_Average_Order',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_61_On_New500_From_20180802_To_20180903':'Transaction_alpha101_61',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_25_On_New500_From_20180802_To_20180903':'Transaction_alpha101_25',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_32_On_New500_From_20180802_To_20180903':'Transaction_alpha101_32',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_Corr_Adjusted_Returns_On_New500_From_20180802_To_20180903':'Transaction_Corr_Adjusted_Returns',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_delta_VOL_Adjusted_Returns_On_New500_From_20180802_To_20180903':'Transaction_delta_VOL_Adjusted_Returns',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_price_skewness_On_New500_From_20180802_To_20180903':'Transaction_price_skewness',
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_ZLJC_On_New500_From_20180802_To_20180903':'Transaction_ZLJC'
			   }

data_dict_1 = {'F:/TOPT_08_09_Validate/Test_Transaction_ACD_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_OLD_UOS_On_ZZ500__From_20180802_To_20180903':['nperiod:12'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_UOS_On_ZZ500__From_20180802_To_20180903':['nperiod:9'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_VR_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_OLD_VR_On_ZZ500__From_20180802_To_20180903':['nperiod:9'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_WAD_On_ZZ500_From_20180802_To_20180903':['nperiod:9'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_KDJ_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_RSV_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Tran_Price_Change_Vol_On_ZZ500__From_20180802_To_20180903':['nperiod:20'],
			   'F:/TOPT_08_09_Validate/Test_Mid_Change_Origin_On_ZZ500__From_20180802_To_20180903':['nperiod:60'],
			   'F:/TOPT_08_09_Validate/Test_VRSI_On_ZZ500__From_20180802_To_20180903':['nperiod:15'],
			   'F:/TOPT_08_09_Validate/Test_RSI_TA_On_ZZ500__From_20180802_To_20180903':['nperiod:15'],
			   'F:/TOPT_08_09_Validate/Test_BIAS_On_ZZ500__From_20180802_To_20180903':['nperiod:15'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_CYM_On_ZZ500__From_20180802_To_20180903':['nperiod:15'],
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_CYS_On_ZZ500__From_20180802_To_20180903':['nperiod:3', 'nperiod:15'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_CHO_On_ZZ500__From_20180802_To_20180903':['nperiod:6'],
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_1_New_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_Sum_Vol_decay_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_Net_DIFF_On_ZZ500__From_20180802_To_20180903':['nperiod:10'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_EMV_On_ZZ500__From_20180802_To_20180903':['nperiod:5'],
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_1_decay_On_ZZ500__From_20180802_To_20180903':['nperiod:20'],
			   'F:/TOPT_08_09_Validate/Test_Tran_Type_Num_Diff_On_ZZ500__From_20180802_To_20180903':['nperiod:10', 'nperiod:40'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_UDL_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_Returns_On_ZZ500__From_20180802_To_20180903':['nperiod:9'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_DMI_no_abs_On_ZZ5002_From_20180802_To_20180903':['nperiod:9'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_AMV_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_WVAD_On_ZZ500__From_20180802_To_20180903':['nperiod:8'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_VPT_On_ZZ500__From_20180802_To_20180903':['nperiod:8'],
			   'F:/TOPT_08_09_Validate/Test_Ask_Bid_AMV_On_ZZ500__From_20180802_To_20180903':['nperiod:3'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_UOS_On_ZZ500__From_20180802_To_20180903/target':['MidReturn60'],
			   'F:/TOPT_08_09_Validate/Test_Transaction_Average_Order_On_ZZ500__From_20180802_To_20180903':['nperiod:10'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_61_On_New500_From_20180802_To_20180903':['nperiod:8'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_25_On_New500_From_20180802_To_20180903':['nperiod:12'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_alpha101_32_On_New500_From_20180802_To_20180903':['nperiod:3'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_Corr_Adjusted_Returns_On_New500_From_20180802_To_20180903':['nperiod:8'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_delta_VOL_Adjusted_Returns_On_New500_From_20180802_To_20180903':['nperiod:12'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_price_skewness_On_New500_From_20180802_To_20180903':['nperiod:12'],
			   'E:/TOPT_1203_Train_add_validate/Test_Transaction_ZLJC_On_New500_From_20180802_To_20180903':['nperiod:5']
			   }

path_dict_2 = {'F:/TOPT_08_09_Validate/Test_Order_Average_Order_On_ZZ500__From_20180802_To_20180903':'Order_Average_Order',
             'F:/TOPT_08_09_Validate/Test_Order_Direction_Amount_decay_On_ZZ500__From_20180802_To_20180903':'Order_Direction_Amount_decay',
             'F:/TOPT_08_09_Validate/Test_Transaction_Order_Percent_Diff_On_ZZ500__From_20180802_To_20180903':'Transaction_Order_Percent_Diff',
             'F:/TOPT_08_09_Validate/Test_Transaction_Order_Times_Diff_On_ZZ500__From_20180802_To_20180903':'Transaction_Order_Times_Diff'} 



data_dict_2 = {'F:/TOPT_08_09_Validate/Test_Order_Average_Order_On_ZZ500__From_20180802_To_20180903':['nperiod:10', 'nperiod:40'],
               'F:/TOPT_08_09_Validate/Test_Order_Direction_Amount_decay_On_ZZ500__From_20180802_To_20180903':['nperiod:10'],
               'F:/TOPT_08_09_Validate/Test_Transaction_Order_Percent_Diff_On_ZZ500__From_20180802_To_20180903':['nperiod:10', 'nperiod:40'],
               'F:/TOPT_08_09_Validate/Test_Transaction_Order_Times_Diff_On_ZZ500__From_20180802_To_20180903':['nperiod:10']}


feature_list1 = list()
feature_name_list1 = list()
caled_list = []
for path in data_dict_1:
	feature_name = path_dict_1[path]
	if feature_name in caled_list:
		pass
	else:
		for period in data_dict_1[path]:
			name = feature_name + '_' + period
			if path == 'F:/TOPT_08_09_Validate/Test_Transaction_UOS_On_ZZ500__From_20180802_To_20180903/target':
				path_file = path + '/' + 'Transaction_UOS_target.h5'
			else:
				path_file = path + '/' + feature_name + '.h5'
			temp_data = pd.read_hdf(path_file, key=period, mode='r')
			valid_points = np.isfinite(temp_data).values.sum()
			print(feature_name, ' ', period, ' feature_length: {}'.format(len(temp_data)), ' valid points: {}'.format(valid_points))


for path in data_dict_2:
	feature_name = path_dict_2[path]
	if feature_name in caled_list:
		pass
	else:
		for period in data_dict_2[path]:
			name = feature_name + '_' + period
			path_file = path + '/' + feature_name + '.h5'
			temp_data = pd.read_hdf(path_file, key=period, mode='r')
			valid_points = np.isfinite(temp_data).values.sum()
			print(feature_name, ' ', period, ' feature_length: {}'.format(len(temp_data)), ' valid points: {}'.format(valid_points))
