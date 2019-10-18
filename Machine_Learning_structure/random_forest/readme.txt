说明：
1.该sample脚本在sklearn 版本号0.20.1 下运行正常
2.关于data_sample.h5的构建： 本sample中design matrix为3列DataFrame（其中两列为feature, 
  另一列为target）feature和target以column names进行区分。该Dataframe以key='X'存入data_sample.h5中
3.sample data 本身均为随机生成序列，均为独立随机变量，因此模型预测效果不存在任何参考性，只用于展示如何使用该模型进行训练