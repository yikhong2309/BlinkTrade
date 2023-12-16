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
from pandas_ta.core import adx, cci, macd, rsi, obv, vwap, atr, bop, ohlc4
from strategy_utils import co_diff_target_cols, useful_X_Cols, origin_cols, Z_cols



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
        # todo 调试
        y_rise_prob = y_predict[0][2]
        y_fall_prob = y_predict[0][0]
        return y_rise_prob, y_fall_prob


class Features_Calculator(object):
    def __init__(self):
        self.origin_cols = origin_cols
        self.Z_cols = Z_cols
        self.co_diff_target_cols = co_diff_target_cols  # 差异特征的目标列
        self.useful_X_Cols = useful_X_Cols  # 最终使用到的因子集合
        return

    def add_all_standalone_features(self, data_df):
        '''
        计算全部技术指标：
            1. 原始指标（部分不会直接用到）
            2. 原始指标的 theta 指标 （归一化）
            3. 原始指标的 diff 指标
        '''
        raw_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'open_time']
        data_df = data_df[raw_cols].copy()
        factor_df = pd.DataFrame()
        # 1 原始指标
        ### ta 的技术指标
        factor_df = self.add_ta_features(factor_df, data_df)
        ### 1.1 Z_score指标
        factor_df = self.cal_Z_score(factor_df, self.origin_cols, 20)
        ### 1.3 diff 指标
        factor_df = self.calculate_diff(factor_df, self.Z_cols, 2)

        ## 2 采样X和y
        assert data_df.shape[0] == factor_df.shape[0]
        factor_df['close'] = data_df['close']
        return factor_df


    def add_ta_features(self, factor_df, raw_df):
        '''
        计算原始技术指标
        包括：
        trend_adx
        trend_cci
        macd
        momentum_rsi
        volume_obv
        volume_vwap
        volatility_atr
        bop
        ohlc4

        '''
        data_df = raw_df[['open', 'close', 'high', 'low', 'volume']]
        data_df.index = pd.to_datetime(raw_df.open_time, unit='ms')

        temp_factor_df = pd.DataFrame()  # 暂存特征，避免index不一致导致的错误
        temp_factor_df['trend_adx'] = adx(data_df['high'], data_df['low'], data_df['close'])['ADX_14']
        temp_factor_df['trend_cci'] = cci(data_df['high'], data_df['low'], data_df['close'])
        temp_factor_df['macd'] = macd(data_df['close'])['MACD_12_26_9']
        temp_factor_df['momentum_rsi'] = rsi(data_df['close'])
        temp_factor_df['volume_obv'] = obv(data_df['close'], data_df['volume'])
        temp_factor_df['volume_vwap'] = vwap(data_df['high'], data_df['low'], data_df['close'], data_df['volume'])
        temp_factor_df['volatility_atr'] = atr(data_df['high'], data_df['low'], data_df['close'])
        temp_factor_df['bop'] = bop(data_df['open'], data_df['high'], data_df['low'],
                                    data_df['close'])  # (open - close)/(high - low)
        temp_factor_df['ohlc4'] = ohlc4(data_df['open'], data_df['high'], data_df['low'], data_df['close'])

        temp_factor_df.reset_index(drop=True, inplace=True)
        factor_df = pd.concat([factor_df, temp_factor_df], axis=1)
        return factor_df


    def cal_Z_score(self, factor_df, column_lists, span):
        '''Z-score'''

        for column_name in column_lists:
            mean_20 = factor_df[column_name].ewm(span, adjust=False).mean()
            std_20 = factor_df[column_name].ewm(96, adjust=False).std()  # todo 超参数搜索
            factor_df[f'{column_name}_Z'] = (factor_df[column_name].values - mean_20) / std_20

        return factor_df

    def calculate_diff(self, data, column_lists, span):
        for column_name in column_lists:
            # First order difference
            data[f'{column_name}_diff_1'] = data[column_name] - data[column_name].shift(span)
            # Second order difference
            data[f'{column_name}_diff_2'] = data[f'{column_name}_diff_1'] - data[f'{column_name}_diff_1'].shift(span)
        return data


    def save_market_coin_data(self, data_df, coin_name):
        '''
        Step 1 先保存market coin data

        '''
        factor_df = self.add_all_standalone_features(data_df)
        factor_df.columns = factor_df.columns + '_' + coin_name
        if coin_name == 'btc':
            self.btc_factor_data = factor_df
        elif coin_name == 'eth':
            self.eth_factor_data = factor_df
        return


    def get_all_features_add_market_coin(self, data_df):
        '''
            Setp 2 全部计算
            计算全部技术指标
            然后将该货币的指标，以及btc，eth指标进行合并

            return factor_df
        '''

        def combine_features( factor_df, btc_data, eth_data):
            concat_df = pd.concat([factor_df, btc_data, eth_data], axis=1)
            return concat_df

        def add_co_diff_features_market_coin(factor_df):
            '''
            加入互相关关系的因子，也就是coin和btc eth的差值
            Step3 加入互相关因子
            '''
            for feature_name in self.co_diff_target_cols:
                factor_df['co_diff_' + feature_name + "_btc"] = factor_df[feature_name] - factor_df[
                    feature_name + "_btc"]
                factor_df['co_diff_' + feature_name + "_eth"] = factor_df[feature_name] - factor_df[
                    feature_name + "_eth"]
            return factor_df

        factor_df = self.add_all_standalone_features(data_df)
        factor_df = combine_features(factor_df, self.btc_factor_data, self.eth_factor_data)  # 合并
        factor_df = add_co_diff_features_market_coin(factor_df)  # 加入互相关因子
        return factor_df[self.useful_X_Cols]



if __name__ == '__main__':
    from Bots.FutureBot.utils.data_utils import get_market_prices

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
