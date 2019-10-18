"""
Feature Module.
"""
from abc import ABCMeta, abstractmethod
import itertools as it
# import utils
from auxiliaryFunctionsNew1009 import split_datetime, time_delta
import pandas as pd
import numpy as np
# import lobapi
import warnings


class base_feature:

    __metaclass__ = ABCMeta
    param_list = []

    def __init__(self, name, feature_type, params=None):

        self.params = params
        self.name = name
        self.feature_type = feature_type
        self.target_scope = ['MidReturn15', 'MidReturn30', 'MidReturn60'
                             'MidReturn90', 'MidReturn300', 'MidReturn600',
                             'MidReturn900', 'MidReturn1500', 'MidReturn2400']

        if params is not None:
            for param_name in params:
                assert param_name in self.__class__.param_list, 'Invalid parameter: ' + str(param_name)

        if len(self.params) != len(set(self.__class__.param_list)):
            warnings.warn("Initialization doesn't cover all parameters.", RuntimeWarning)
        self.parameters_comb()


    def parameters_comb(self):

        # return a list of all params combinations
        # for orderbook features, nperiod should always put at the first place!
        if len(self.__class__.param_list) > 0:
            params_combs = list(it.product(*(self.params[p] for p in self.__class__.param_list)))
            params_comb_name_base = ':{} '.join(self.__class__.param_list)
            params_comb_name_base += ':{}'
            params_comb_name = [params_comb_name_base.format(*t) for t in params_combs]
            self.params_comb_names, self.params_combs = params_comb_name, params_combs
            if len(self.__class__.param_list) == 0:
                self.feature_columns = self.name
            else:
                self.feature_columns = self.params_comb_names


    # def compute_feature_tick_data_only(self, data_feed):

    #     # using only tick data (no order book data the calulate features)

    #     required_idx = data_feed['tick_data'].index
    #     if len(required_idx) > 0:
    #         result = pd.DataFrame()
    #         if len(self.__class__.param_list) == 0:
    #             result[self.feature_columns] = self.feature(data_feed).loc[required_idx]
    #             return result
    #         elif len(self.__class__.param_list) > 0:
    #             for i in range(len(self.feature_columns)):
    #                 result[self.feature_columns[i]] = self.feature(data_feed, *self.params_combs[i])[required_idx]
    #             return result
    #     else:
    #         return None


    def compute_feature(self, data_feed, data_affiliated_status):

        # using only tick data (no order book data the calulate features)

        required_idx = data_affiliated_status.index

        if len(self.feature_type) == 1 and 'tick' in self.feature_type:
            if len(required_idx) > 0:
                result = pd.DataFrame()
                if len(self.__class__.param_list) == 0:
                    result[self.feature_columns] = self.feature(data_feed).reindex(required_idx)
                    return result
                elif len(self.__class__.param_list) > 0:
                    for i in range(len(self.feature_columns)):
                        result[self.feature_columns[i]] = self.feature(data_feed, *self.params_combs[i]).reindex(required_idx)
                    return result
            else:
                return None
        else:
            if len(required_idx) > 0:
                result = pd.DataFrame()
                if len(self.__class__.param_list) == 0:
                    temp_result = self.feature(data_feed)
                    if len(temp_result) > 0:
                        low_bound = np.max([temp_result.index[0], required_idx[0]])
                        up_bound = np.min([temp_result.index[-1], required_idx[-1]])
                        target_index = required_idx[(required_idx > low_bound) & (required_idx < up_bound)]
                        new_temp_result = ~temp_result.index.duplicated(keep='last')
                        no_duplicated = temp_result.iloc[new_temp_result]
                        pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
                        result_feature = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index).reindex(required_idx)
                        result[self.feature_columns] = result_feature
                    else:
                        result[self.feature_columns] = temp_result.reindex(required_idx)
                    return result
                elif len(self.__class__.param_list) > 0:
                    for i in range(len(self.feature_columns)):
                        temp_result = self.feature(data_feed, *self.params_combs[i])
                        if len(temp_result) > 0:
                            low_bound = np.max([temp_result.index[0], required_idx[0]])
                            up_bound = np.min([temp_result.index[-1], required_idx[-1]])
                            target_index = required_idx[(required_idx > low_bound) & (required_idx < up_bound)]
                            new_temp_result = ~temp_result.index.duplicated(keep='last')
                            no_duplicated = temp_result.iloc[new_temp_result]
                            pos_new_result = no_duplicated.index.searchsorted(target_index, side='left')
                            result_feature = pd.Series(no_duplicated.iloc[pos_new_result].values, index=target_index).reindex(required_idx)
                            result[self.feature_columns[i]] = result_feature
                        else:
                            result[self.feature_columns[i]] = temp_result.reindex(required_idx)
                    return result
            else:
                return None




    @abstractmethod
    def feature(self, *args):
        """
        This is the thing to override. Researcher should define their logic of
        computing feature with this function. Notice that this function should
        return a Pandas DateFrame.
        """
        pass
