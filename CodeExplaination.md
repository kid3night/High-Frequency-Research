Chinese A Share High Frequency Research



Code 说明文档

 

Part 1. 

数据清洗部分：./ raw_data_clean/

\1.   data_split.py 

作用:将从数据库中导出的.csv 文件转化为.h5文件

使用方法:将data_split.py 中的if __name__ == "__main__":部分下的

data_file_path = "F:/qtdata/"  csv数据的存放位置

data_type = 'tick'    要split的数据类型（tick, transaction, order, index）

save_path = 'F:/{}_raw'.format(data_type)  对应split后数据的输出路径

bgd = "20180824"  开始日期

edd = "20180904"  结束日期

multi_proc = 4   使用几个进程,这里默认4个,可以在内存可以容忍时使用更多

  程序的运行结果可以参考F:\tick_raw, F:\transaction_raw, F:\order_raw

 

\2.   tick_bar_generator_class.py

作用:将获取的tick_raw数据转换成对应的tick_bar数据并生产对应的target_bar数据

\3.   使用方法:将tick_bar_generator_class.py中的if __name__ == "__main__":部分下的

h5_path = 'F:/tick_raw'  对应的tick_raw数据存放地址

tick_bar = 'F:/tick_bar'  输出的tick_bar数据存放地址

targets_bar = 'F:/targets_bar_new_add' 输出的target_bar数据存放地址

bgd = "20180830"  开始日期

edd = "20180831"  结束日期

freq = "3S"   转换成几秒的bar

multi_proc = 0  使用几个进程

(其他参数无用可以忽略)

另外在该脚本的line 162中

TargetSeriesGenerate_NEW(data, rt_series=[15, 60, 120, 180, 300])

rt_series = [15, 60, 120, 180, 300] 是用于定义target的未来秒数, 这里将计算未来15s, 60s, 120s, 180s, 300s 以及15~60s, 60~120s, 120~180s, 18~300s的mid return. 可以根据需要修改这里的rt_series的数值

 

\4.   h5concat.py

作用:将成的tick_bar, target_bar, transaction_raw, order_raw 数据concat成比较长的flexible h5 file.

\5.   使用方法:将h5concat.py中的if __name__ == "__main__":部分下的

h5_path = 'F:/tick_bar' 对应的tick_bar数据地址

data_type = 'tick'  对应想concat的数据类型（tick, transaction, order, index）

save_path = 'F:/{}_concated_data_check_08_09'.format(data_type) 输出结果存放路径

save_name = '{}_concated.h5'.format(data_type) 输出结果名

concat_mode = 'from_begin' 两种concat方式(从给定的开始结束日期直接concat成一个独立的h5 file, 如果使用’other’则会往现有的h5 file中继续concat文件

begin_date = 20180802 concat开始日

end_date = 20180905 concat 结束日 (由于文件过大可能会导致h5 file出现问题 不建议concat长度超过2个月)

 

依次运行完毕以上的三种文件后, 正常情况可以获得: transaction_concated.h5, tick_concated.h5, order_concated.h5, targets_concated.h5 这四个文件为在进行feature 构建时的必要元素. 实例路径为F:\targets_concated_data_check, F:\order_concated_data_check, F:\tick_concated_data_check, F:\transaction_concated_data_check

 

Part2. 

feature 构建部分./TANG_FINISHED_PNL/

feature测试框架的主要组件在./ TANG_FINISHED_PNL/backtest_main_files/下

 

\1.   baseSignal_Modified 定义了basefeature的抽象类, 所有feature函数均继承自该base

主要feature函数均存放在./ TANG_FINISHED_PNL/backtest_main_files/feature_files/下

以feature_selected.py中的Transaction_ACD为例: 定义一个新feature时, 需要新建一个class, class 名即为feature名, 另外需要注意:

from baseSignal_Modified import base_feature 

class xxxx(base_feature) 新的feature是继承自base_Signal_Modified.py /base_feature class的

\2.   在选择合适的feature name之后, 需要在param_list = ['nperiod']中定义feature的参数名列表, 大多数情况下只选择一个参数,即为nperiod (每个时刻往前取的tick数量)

在确定好param_list后, 需要实际构建feature function

def feature(self, data_fed, p1, p2, p3, …) 这里p1, p2, p3均应该是在param_list 中按顺序定义的参数名

\3.   在feature function的定义过程中首先会从data_fed中获取数据(data_fed为一个dictionary, 具体包含哪些数据由feature定义者给定的configfile中的参数确定. 在Transaction_ACD这个feature中, 只用到了transaction_data. 在经过feature function的计算后, 每个featurefile应当return一个pandas series 序列作为该feature的计算结果(order, transaction数据算出来的结果index和tick不同这个测试框架会自动处理对齐补全) 

可以在一个feature中选择多个数据源进行计算如feature_selected.py line 641 Transaction_Order_Percent_Diff 这个feature同时使用了tick, transaction, 和order数据但是需要注意时间戳的处理问题

\4.   Configfile的构建: 在构建一个feature的时候需要考虑测试什么universe的数据, 测试的时间长度, 使用的数据种类, feature参数的参数空间定义, feature的输出地址等, 本框架使用.json完成. 以./TANG_FINISHED_PNL/config_feature_selected/Transaction_ACD.json为例:

Import uniserse:1 代表使用外部csv/xls文件import universe, 设置为0表示不从外部import, 会在self_specified_universe中自定义一个universe list

如果确定自定义则在universe_file_name 中写出universe csv的地址(现统一使用一列Series的universe file, 示例可查看./ TANG_FINISHED_PNL/universe_ZZ500.csv.

 

test_result_save_path 为feature计算结果的输出地址

 

feature_file_name 为feature file名称, 必须保持和feature的脚本名一致,程序用该种方法定位到具体的feature file

feature_name为feature名称, 必须保持和feature 的class名一致,程序用该种方法定位到具体的feature function

test_time_start 测试的起始日期

test_time_end 测试的结束日期

 

not_plot_pnl 是否对该feature进行交易信号转换并画出单个feature转换成交易信号之后的pnl曲线

not_plot_pnl = 0 则计算corr, 画出feature单调性曲线柱状图, 并且计算pnl曲线

not_plot_pnl = 1 则计算corr, 画出feature单调性曲线柱状图, 不计算pnl曲线

not_plot_pnl = 2则只计算pnl曲线

multi_process_feature 计算feature过程多进程控制默认0为单进程,一般选择10

 

multi_process_result feature计算完后backtest过程多进程控制. 一般根据feature的参数量确定, 如果5个参数则选择5 (注意次过程内存消耗有些大, 不能选择太多)

 

feature_params 根据feature class 中定义的params_list确定 每个参数要测多个数值 用dictionary of list 表示

 

feature_type 计算该feature使用的数据类型可以同时写出多个, 这样在feature 的data_fed中就能够提取到具体的数据

 

each_type_data_need 因为计算feature只需要每种data的部分column, 使用该选项可以定义从每种data中选择哪些column放入data_fed.

 

generated_feature_h5_path, generated_target_h5_path 这两个变量在使用已经生成的feature file进行回测时使用, 前者为计算好的feature file名,后者为计算好的target file名.

new_result_subfolder_name 为feature backtest输出结果创建一个subfolder, 在这里定义其名称.

operation_type 定义这个config的计算目的:

operation_type = full_backtest 表示先计算feature 然后使用计算的feature进行 backtest

operation_type = feature_calculation表示 仅仅计算feature

operation_type = others/anythingother than full_backtest or feature_calculation表示仅仅根据现有feature file进行backtest 此时上两个变量就会发挥作用

 

\5.   每个feature的每次test都要定义一个configfile来运算,可以选择直接修改老的configfile或者新建configfile. configfile的名称应当与feature name相同

\6.   运行测试框架需要使用 

./TANG_FINISHED_PNL/config_executor.py 

使用该文件时需要修改line 24行的configfile储存地址为对应地址在命令行中使用

python config_executor.py config_executor.py -c xxxx 其中xxxx即为feature name

或者定义类似的batch test脚本 批量测试多个feature  

./TANG_FINISHED_PNL/batch_configs_executor.py

./TANG_FINISHED_PNL/batch_config_executor.py

 

\7.   其他注意事项:

数据来源的地址确定:

./TANG_FINISHED_PNL/backtest_main_files/backtest_engine_final.py

Line 40 ~ line 48 直接修改self. tick_path, self. target_path, self. order_path

self.transaction_path 实例code中可以看到40~43表示的为6~8月的数据 45~48表示的为8~9月的数据

同时在改数据来源的时候也得修改

./TANG_FINISHED_PNL/backtest_main_files/get_universe.py中的line 94, 96, 103, 105将对应的数据和上面的数据地址对应一致

 

Backtest 信号转换选择的分位点 比如达到整体分为点的98进,-90出 应该在

./TANG_FINISHED_PNL/backtest_main_files/backtest_engine_final.py 中line 26的 upper, lower的数值修改成想要的分位点

同时需要在

./TANG_FINISHED_PNL/backtest_main_files/auxiliaryFunctionsNew1009.py的line 317将0.0015 和0.0005修改为 upper, 和lower (默认情况使用的是绝对数值0.0015 和0.0005作为信号转换数值点) 调整数值也可以在这里直接调整

 

 

Part 3

Machine Learning Part

 

\1.   Data combination

在通过feature测试框架完成feature的测试以及筛选后, 可以使用各类machine learning 开源框架进行machine learning model的调试与构建.

在构建模型之前, 需要先构建设计阵和清洗数据以保证每个input的sample都是有效的sample.

./ Machine_Learning_structure/data_construct_sample_p1.py

在确定好要include的feature的feature路径,名称以及对应的参数组合后, 应当写一个如以上文件格式的py file然后运行以观察每个入选的feature向量的总长度, 有效点长度. (如果有效点长度过短,则应当删除该feature) 这里的sample file之所有分为dict1, dict2两块是因为dict1没有使用到order数据, 所以为500整个universe的数据, 而dict2用到了order数据所以为500支股票中SZ市场的数据. dict1, dict2分别对应的所有feature的长度应当一样长 且dict2 feature的长度应当为dict1 feature长度的约一半. 进一步确认完毕后,使用

./ Machine_Learning_structure/data_construct_sample_p2.py

(注意在该文件的line 115 和line134 将交集部分的长度填上 该长度可由p1.py的输出结果得到

对feature file进行combine (由于pandas df/series 的concat速度非常慢(每次concat需要对齐index, 在index很长的情况下效率极低), 故使用numpy.concatenate() 先将所有的feature values提取出来再进一步拼接. 注意 dict的feature部分恰巧对应dict1 feature的前半段, 故使用dict1_feature.iloc[:len(dict2_feature)] 即可提取出dict1 feature的对应时间区间(可以找部分feature进行确认)

由该文件得到了对应的feature matrix 之后即可放入提交的machine learning 框架进行model training.

 

\2.   Model training 

使用./ Machine_Learning_structure/lightgbm

或者./ Machine_Learning_structure/random_forest 配上生成的feature_matrix.h5 均可进行training

 

\3.   Modeling test set validation and backtest

在完成model的训练之后, 需要将样本外的数据拿出对比. 在使用同样的流程构建好样本外的feature_matrix之后 使用model.predict() 即可得到样本外的model prediction y_hat.

由于同时还需要对应区间的stock spread rate的信息来对prediction数值进行进一步操作(y_hat – sread_rate) 所以需要调用

./TANG_FINISHED_PNL/config_ML/Ask_Bid_Spread_rate.json 

对feature Ask_Bid_Spread_rate 得到对应所有股票的spread_rate 序列

计算完毕后, 修改并使用feature_spread_operation.py将结果构建为feature形式(这里需要构建成feature形式的具体原因就是可以直接使用feature的测试框架直接bt, 所以需要将计算出来的信号”伪装成一个feature”. 完成后可以修改部分

./ TANG_FINISHED_PNL/config_ML/SZ_feature.json的参数(将路径改为新生成的信号路径)

 

 

Part 4

Feature files 

使用的feature files放在./TANG_FINISHED_PNL/feature_files/feature_selected_1220.py中

新的feature formula为./Feature_Formulas_NEW.txt

 

关于F:/ 文档说明

TOPT_1030 存放backtest的结果

TOPT_1101 , TOPT_08_09_Validate, TOPT_08_09_Validate_add 存放的为feature在500 universe上的计算结果

其中TOPT_1101为6~8月，后两者为8~9月 TOPT_08_09_Validate_add为TOPT_08_09_Validate的补充

TOPT_1203_Train， TOPT_1203_Test 存放的为feature在1800 universe上的计算结果，其中前者为6~8月，后者为8~9月

 

Backtest示例说明：

F:/TOPT_1030下

Test_SZ_feature_On_TrainSet_Train_ADD_features_1218_minus_spread_add_From_20180601_To_20180801

Test_SZ_feature_On_TestSet_Train_ADD_features_1218_minus_spread_add_From_20180802_To_20180903

这两个文件夹中分别存放的是500 universe 中 SZ股票数据在lightgbm model训练后得到的回测结果，前者为训练集部分，后者为测试集部分。

对应的参数组合和训练文件可以参考：./model_1218/new_trained_model_1218.xlsx, 

./model_1218/lgb_model_1218.ipynb

在该model test result 的nperiod 1, 2, 3 分别对应的是该model的三个参数组合

 

Test_SZ_feature_On_Test_predict_ret_minus_spread_1800_From_20180601_To_20180801和

Test_SZ_feature_On_TESTSET_prediction_minus_spread_1800_From_20180802_To_20180903

这两个文件夹中分别存放的是1800 universe 中 SZ股票数据在lightgbm model训练后得到的回测结果，前者为训练集部分，后者为测试集部分。

对应的参数组合和训练文件可以参考：./model_1218/model_1203_parameters.xlsx, 

./model_1218/1800_universe_train.ipynb 只测试了一个model, nerpiod 10 即为该model的测试结果

 

 

以上的测试结果为(predicted 60s return – spread rate) 然后以0.0015 和 0.0005作为分割点转化为交易信号进行交易

 

 