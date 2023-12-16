import pandas as pd
import numpy as np
import logging
import time
import sys
sys.path.append("..")
from Bots.FutureBot.strategies.reverse_detector import Reverse_Detector, Features_Calculator
from Bots.FutureBot.strategies.strategy_utils import np_round_floor
from Bots.FutureBot.utils.trade_utils import Order_Structure
from Bots.FutureBot.utils.information import Info_Controller


class StrategyInterface(object):
    def __init__(self ):
        pass

    def get_order_list(self, info_controller:Info_Controller,):
        '''输入数据，得到交易订单'''
        pass

    def get_order(self):
        pass


class Strategy_mean_reversion(StrategyInterface):
    '''均值复归策略'''
    def __init__(self):
        super().__init__()
        self.reverse_detector = Reverse_Detector(
            model_save_path='/home/ec2-user/HengTrader/Bots/FutureBot/models/',
            model_name='rf_3class_70_30.pkl'
        )
        self.features_calculator = Features_Calculator()


    def get_theta(self, data_df):
        Boll_df = pd.DataFrame(index=data_df.index)
        # todo 修改计算方法 ohlc4_Z
        Boll_df['mean_20'] = data_df[['close']].ewm(span=20, adjust=False).mean()
        Boll_df['std_20'] = data_df[['close']].ewm(span=96, adjust=False).std()
        Boll_df['close'] = data_df['close']

        Boll_df.dropna(inplace=True)

        # 计算偏离度 theta = (p - ma) / sigma
        Boll_df['theta'] = (Boll_df['close'] - Boll_df['mean_20']) / Boll_df['std_20']
        return Boll_df['theta'].values[-1]


    def update_info(self, info_controller):
        '''
            1. 对于info_controller的信息进行更新
        '''
        data_dict = info_controller.strategy_info.price_dict
        candidate_symbols = info_controller.strategy_info.candidate_symbols

        # 1. 更新theta信息
        for symbol in candidate_symbols:
            data_df = data_dict[symbol]
            theta = self.get_theta(data_df)
            info_controller.strategy_info.exchange_info_df.loc[symbol,"theta"] = np.round(theta,4)

        return info_controller


    def get_order_list(self, info_controller:Info_Controller, target_position=1.):

        order_list = []
        # 1. 更新 inforcontroller 的 ishold 信息
        info_controller = self.update_info(info_controller)

        data_dict = info_controller.strategy_info.price_dict
        exchange_info_df = info_controller.strategy_info.exchange_info_df

        # 2. 更新并且记录btc 和 eth的features
        self.features_calculator.save_market_coin_data(data_dict["BTCUSDT"], coin_name="btc")
        self.features_calculator.save_market_coin_data(data_dict["ETHUSDT"], coin_name="eth")

        # 1. 判断是否持有
        held_set, un_held_set = info_controller.get_symbols_held_sets()

        # 对于持有的币种，进行判断是否需要平仓
        for symbol in held_set:
            temp_order = self.held_set_logic(symbol, info_controller)
            if temp_order:
                order_list.append(temp_order)
                info_controller.strategy_info.update_order_info(
                    symbol=temp_order.symbol,
                    side=temp_order.side,
                    order_time=time.time()
                )

        # 对于未持有的币种，进行判断是否需要开仓
        for symbol in un_held_set:
            temp_order = self.un_held_set_logic(symbol, info_controller)
            if temp_order:
                order_list.append(temp_order)
                info_controller.strategy_info.update_order_info(
                    symbol=temp_order.symbol,
                    side='HOLD',
                    order_time=time.time()
                )

        return order_list


    def held_set_logic(self, symbol, info_controller:Info_Controller):
        '''
        对于持有的币种，进行判断是否需要平仓
        1. 判断止损
        2. 判断theta

        return order
        '''
        # 1. 判断是否需要止损
        unrealizedProfit = info_controller.account_info.position_df.loc[symbol, "unrealizedProfit"]
        if unrealizedProfit < -2:  # todo 加入设置中
            order = self.get_close_order(symbol, info_controller=info_controller)
            return order

        # 按照时间，判断是否需要平仓
        judge_close = self.judge_close(symbol, info_controller)
        if judge_close:
            order = self.get_close_order(symbol, info_controller=info_controller)
            return order
        else:
            return None


    def un_held_set_logic(self, symbol, info_controller):
        '''
        未持有的币种，进行判断是否需要开仓
        '''
        # 1. 判断theta
        theta = info_controller.strategy_info.exchange_info_df.loc[symbol, "theta"]
        if np.abs(theta) > 0.7:
            # 结合机器学习，判断是否需要开仓
            side = self.judge_open_side(symbol, info_controller)
            if side == 'HOLD':
                return None
            else:
                order = self.get_open_order(symbol, side=side, info_controller=info_controller)
                return order
        else:
            return None


    def get_open_order(self, symbol, side, info_controller):
        '''
            开仓逻辑，由ml判断是否开仓，以及开仓的方向
            return order
        '''
        def cal_buy_quantity(info_controller, price):
            '''
            计算买入量
            '''
            balance = info_controller.account_info.USDT_value
            if balance > 1000:
                quantity = 100. / price
            elif balance > 500:
                quantity = 50. / price
            else:
                quantity = 0.
            return quantity

        order = Order_Structure()
        order.symbol = symbol
        order.side = side
        # 获取挂单价格 price
        order.price = info_controller.get_price_now(symbol)
        # 计算买入量 quantity
        quantity = cal_buy_quantity(info_controller, order.price)
        if quantity == 0:
            return None
        # 保留小数点位数
        stepDecimal = info_controller.strategy_info.exchange_info_df.loc[symbol, "quantityPrecision"]
        order.quantity = np_round_floor(quantity, stepDecimal)
        return order


    def get_close_order(self, symbol, info_controller):
        '''
            close 平仓。
            交易方向总是和持仓方向相反
        '''
        order = Order_Structure()
        order.symbol = symbol

        positionAmt = info_controller.account_info.position_df.loc[symbol, "positionAmt"]
        stepDecimal = info_controller.strategy_info.exchange_info_df.loc[symbol, "quantityPrecision"]

        if order.quantity < 0:
            order.quantity = -1 * positionAmt
        else:
            order.quantity = positionAmt
        order.quantity = np_round_floor(order.quantity, stepDecimal)

        # 判断交易方向，总是和持仓方向相反
        if positionAmt > 0:
            order.side = 'SELL'
        elif positionAmt < 0:
            order.side = 'BUY'

        # 获取挂单价格
        order.price = info_controller.get_price_now(symbol)

        try:
            order.selfcheck()
        except AssertionError as e:
            logging.info('-----------------AssertionError-----------------------')
            logging.info(f"AssertionError:{e}".format(order.symbol))
            logging.info('------------------------------------------------------')
            return None

        return order


    def judge_open_side(self, symbol, info_controller) -> str:
        '''
        判断是否需要开仓
        return side
            - 'BUY'
            - 'SELL'
            - 'HOLD'
        '''
        side = self.get_ml_trade_derection(symbol, info_controller)
        return side


    def judge_close(self, symbol, info_controller) -> bool:
        '''
        判断是否需要平仓
        '''
        order_side = info_controller.strategy_info.order_info_df.loc[symbol, 'side']
        order_time = info_controller.strategy_info.order_info_df.loc[symbol, 'order_time']

        # 计算时间差
        time_now = pd.to_datetime(time.time(), unit='s')
        time_difference = time_now - order_time  # 是一个timedelta对象
        # 检查差值是否大于300分钟
        is_greater_than_300_minutes = time_difference > pd.Timedelta(minutes=300)

        if is_greater_than_300_minutes:  # 超过300分钟就平仓
            if order_side == 'HOLD':
                return False  # 说明这个平仓信号已经发出过
            else:
                return True
        else:
            return False


    def get_ml_trade_derection(self, symbol, info_controller):
        '''
        由机器学习判断交易方向
        '''
        y_rise_prob, y_fall_prob  = self.get_ml_prediction(symbol, info_controller)

        logging.info('--------------------judge_buy---------------------------')
        logging.info("symbol:{}".format(symbol))
        logging.info("ml_pred:{}".format(y_fall_prob))
        logging.info("ml_pred:{}".format(y_rise_prob))
        logging.info('---------------------------------------------------------')

        # todo 调整合适的阈值，观察是否会有二者同时开仓的情况
        if y_rise_prob > 0.4392:
            return 'BUY'
        elif y_fall_prob > 0.4535:
            return 'SELL'
        else:
            return "HOLD"


    def get_ml_prediction(self, symbol, info_controller):
        '''获取机器学习的预测结果'''
        price_df = info_controller.strategy_info.price_dict[symbol]
        factor_df = self.features_calculator.get_all_features_add_market_coin(price_df)
        factor_df = factor_df[self.features_calculator.all_X_cols]
        y_fall_prob, y_rise_prob = self.reverse_detector.get_machine_learning_pridictions(factor_df)
        return y_fall_prob, y_rise_prob



class Hedge_Strategy(StrategyInterface):
    '''
    对冲策略
    暂停使用
    '''
    def __init__(self):
        super().__init__()
        pass

    def get_order_list(self, info_controller:Info_Controller):
        order_list = []

        # 1. 获取position
        position_df = info_controller.account_info.position_df
        # 2. 计算对应的notion value
        usdt_notional_value = position_df['notional'].sum()
        # 3. 根据notion value 计算对应的BTCUSDT买卖量
        print(usdt_notional_value)
        quantity = -1 * usdt_notional_value

        btc_order = self.get_order(
            symbol='BTCUSDT',
            quantity=quantity,
            info_controller=info_controller)

        order_list.append(btc_order)
        return order_list

    def get_order(self, symbol, quantity, info_controller):
        '''
            开仓逻辑，由ml判断是否开仓，以及开仓的方向
            return order
        '''
        order = Order_Structure()
        order.symbol = symbol

        if quantity > 0:
            order.side = 'BUY'
        elif quantity < 0:
            order.side = 'SELL'
        else:
            return None
        # 获取挂单价格 price
        order.price = info_controller.get_price_now(symbol)
        # 计算买入量 quantity
        quantity = np.abs(quantity) / order.price
        # 保留小数点位数
        stepDecimal = info_controller.strategy_info.exchange_info_df.loc[symbol, "quantityPrecision"]
        order.quantity = np_round_floor(quantity, stepDecimal)

        return order

