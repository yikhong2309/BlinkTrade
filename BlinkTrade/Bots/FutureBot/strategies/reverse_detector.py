"""
File: reverse_detector.py
Author: Henry Miao
Date Created: 2023-10-03
Last Modified: --
Description: Use machine learning to detect reversion of price.
"""
import pandas as pd
import numpy as np
import pickle
import pandas_ta



class Reverse_Detector(object):
    def __init__(self, model_save_path, model_name):
        self.ml_model = self.load_ml_model(model_save_path, model_name)

    def load_ml_model(self, model_save_path, model_name):
        with open(model_save_path + model_name, 'rb') as f:
            model = pickle.load(f)
        return model

    def get_machine_learning_pridictions(self, factor_df):
        '''
        :param factor_df: 传入的数据，包含了所有的因子
        '''
        data_df = factor_df.copy()
        y_predict = self.ml_model.predict_proba(data_df.iloc[-1:, :])
        y_rise_prob = y_predict[0][2]
        y_fall_prob = y_predict[0][0]
        return y_rise_prob, y_fall_prob


class Features_Calculator(object):
    def __init__(self):
        self.funcName_list = ['hl2', 'ohlc4', 'tema', 'rsi', 'wcp', 'cmo', 'bias', 'zlma',
                              'hlc3', 'zscore', 'dm', 'pgo', 'midpoint', 'decreasing',
                              'decay', 'ui', 'aroon', 'cdl_z', 'vortex', 'amat']  # pandas_ta中的函数名

        self.factorName_list = ['HL2', 'OHLC4', 'TEMA_10', 'RSI_14', 'WCP', 'CMO_14', 'BIAS_SMA_26',
                                'ZL_EMA_10', 'HLC3', 'ZS_30', 'DMN_14', 'PGO_14', 'MIDPOINT_2', 'DEC_1',
                                'LDECAY_5', 'UI_14', 'AROOND_14', 'close_Z_30_1', 'VTXM_14', 'AMATe_SR_8_21_2']  # 因子名

        # 单个币种的全部特征
        self.X_standalone_cols = self.factorName_list  # 20个

        # 交互特征  10个
        self.co_diff_target_cols = (['DMN_14', 'ZS_30', 'CMO_14', 'RSI_14', 'UI_14', 'PGO_14',
                                     'close_Z_30_1', 'VTXM_14', 'BIAS_SMA_26', 'AROOND_14'])  # 10个
        self.co_diff_target_cols_btc = ['co_diff_' + x + "_btc" for x in self.co_diff_target_cols]
        self.co_diff_target_cols_eth = ['co_diff_' + x + "_eth" for x in self.co_diff_target_cols]

        # btc eth自身的特征  20*2 = 40个
        self.btc_X_cols = [x + "_btc" for x in self.X_standalone_cols]
        self.eth_X_cols = [x + "_eth" for x in self.X_standalone_cols]

        # 所有X_cols
        self.all_X_cols = self.X_standalone_cols + self.co_diff_target_cols_btc + self.co_diff_target_cols_eth + self.btc_X_cols + self.eth_X_cols
        self.useful_X_Cols = self.X_standalone_cols + self.co_diff_target_cols_btc + self.co_diff_target_cols_eth  ## 目前来看更有效的特征子集
        return

    def save_market_coin_data(self, data_df, coin_name):
        '''
        保存单个币种的数据
        '''
        factor_df = self.add_all_standalone_features(data_df)[self.X_standalone_cols]
        factor_df.columns = factor_df.columns + '_' + coin_name
        if coin_name == 'btc':
            self.btc_factor_data = factor_df
        elif coin_name == 'eth':
            self.eth_factor_data = factor_df
        return

    def get_all_features_add_market_coin(self, data_df):
        '''
            计算全部技术指标
            然后将该货币的指标，以及btc，eth指标进行合并

            return factor_df
        '''
        # 1. 计算全部技术指标
        factor_df = self.add_all_standalone_features(data_df)[self.X_standalone_cols]
        # 2. 与保存的btc eth数据进行合并
        factor_df = self.combine_features(factor_df, self.btc_factor_data, self.eth_factor_data)
        factor_df = self.add_co_diff_features_market_coin(factor_df)  # 加入互相关因子
        return factor_df[self.all_X_cols]

    def add_all_standalone_features(self, data_df):
        '''
        计算全部技术指标：
            1. 原始指标（部分不会直接用到）
            2. 原始指标的 theta 指标 （归一化）
            3. 原始指标的 diff 指标
        '''
        raw_cols = ['open', 'high', 'low', 'close', 'quote_volume', 'open_time']
        data_df = data_df[raw_cols].copy()
        # 1 原始指标
        ### ta 的技术指标
        data_df = self.add_ta_features(data_df)
        return data_df

    def add_ta_features(self, raw_df):
        '''
        计算 pandas ta 原始技术指标
        '''
        raw_df.rename({'quote_volume': 'volume'}, axis=1, inplace=True)
        for indicator_func_name in self.funcName_list:  # 遍历全部所需函数，计算对应特征
            fun = getattr(raw_df.ta, indicator_func_name)
            temp_data = fun(append=True)
        return raw_df

    def cal_Z_score(self, factor_df, column_lists, span):
        '''Z-score'''
        for column_name in column_lists:
            mean_20 = factor_df[column_name].ewm(span, adjust=False).mean()
            std_20 = factor_df[column_name].ewm(96, adjust=False).std()  # todo 超参数搜索
            factor_df[f'{column_name}_Z'] = (factor_df[column_name].values - mean_20) / std_20
        return factor_df

    def combine_features(self, factor_df, btc_data, eth_data):
        concat_df = pd.concat([factor_df, btc_data, eth_data], axis=1)
        return concat_df

    def add_co_diff_features_market_coin(self, factor_df):
        '''
        加入互相关关系的因子，也就是coin和btc eth的差值
        '''
        co_diff_target_cols = self.co_diff_target_cols
        for feature_name in co_diff_target_cols:
            factor_df['co_diff_' + feature_name + "_btc"] = factor_df[feature_name] - factor_df[feature_name + "_btc"]
            factor_df['co_diff_' + feature_name + "_eth"] = factor_df[feature_name] - factor_df[feature_name + "_eth"]
        return factor_df



if __name__ == '__main__':
    from BlinkTrade.BlinkTrade.Bots.SpotBot.utils.data_utils import get_market_prices

    btc_price_df = get_market_prices('BTCUSDT', '15m')
    eth_price_df = get_market_prices('ETHUSDT', '15m')

    print(btc_price_df.head())
    f_c = Features_Calculator()

    f_c.save_market_coin_data(btc_price_df, coin_name='btc' )
    f_c.save_market_coin_data(eth_price_df, coin_name='eth' )

    price_df = get_market_prices('AXSUSDT', '15m')

    factor_df = f_c.get_all_features_add_market_coin(price_df)[f_c.useful_X_Cols]

    print(factor_df.head())


    # # 测试预测部分
    model_save_path = '../models/'
    model_name = 'rf_3class_70_30.pkl'
    r_d = Reverse_Detector(model_save_path, model_name)
    y_rise_prob, y_fall_prob = r_d.get_machine_learning_pridictions(factor_df)

    print(factor_df.iloc[-1:, :])
    print('y_rise_prob: ', y_rise_prob)
    print('y_fall_prob: ', y_fall_prob)
