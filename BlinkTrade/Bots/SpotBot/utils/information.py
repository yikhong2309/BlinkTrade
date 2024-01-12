import pandas as pd
from binance.spot import Spot as SpotClient

import sys
sys.path.append(r'C:\Users\Admin\Desktop\\')

from Crypto_Bot.utils.data_utils import get_decimal_precision
from Crypto_Bot.utils.privateconfig import *


## Parent Class
class Info_Interface():
    def __init__(self):
        pass

    def init_info(self, client):
        pass

    def update_info(self, client):
        pass

## Subclass 1
class Account_Info(Info_Interface):
    '''
    Account Info, Current position info & Historical position info
    '''
    def __init__(self, config_dict):

        # The super().__init__() line is calling the __init__ method of the parent class (Info_Interface) within the __init__ method of the Account_Info class.
        super().__init__()

        self.position_df = None
        self.USDT_value = None
        self.account_value = None

    def init_info(self, client):
        '''
        1. init "position_df"
            1. init "balance"
            2. init "is_hold"
            3. init "bid_price" for the holds
        '''
        # Get account info
        account_infomation = client.account()  

        # Account balances info
        position_df = pd.DataFrame(account_infomation['balances'])
        self.USDT_value = float(position_df[position_df["asset"] == "USDT"]['free'].values[0])

        # Get actual price of "XXXUSDT"
        # Get all real market symbols
        exchange_dict = client.exchange_info()
        temp = exchange_dict['symbols']
        market_symbols = set([symbol_dict['symbol'] for symbol_dict in temp])

        # We only need currencies pair with USDT
        # 此处实际上得到了全部的持仓信息，包括了不在候选交易对中的
        fake_symbols = [x + 'USDT' for x in position_df['asset'].values]
        position_df['symbol'] = fake_symbols

        is_symbol_exist = [True if x in market_symbols else False for x in fake_symbols]

        # all symbol with USDT and Exist
        exist_symbols = list(position_df[is_symbol_exist].symbol.values) 
        position_df = position_df[is_symbol_exist]

        price_df = pd.DataFrame(client.ticker_price(symbols=exist_symbols))
        position_df = pd.merge(position_df, price_df, on='symbol', how='left')
        position_df.set_index('symbol', inplace=True, drop=True)

        # Price data Formatting 
        convert_dict = {'free': float,
                        'locked': float,
                        'price': float,
                        }
        position_df = position_df.astype(convert_dict)

        # Calculate if it is already in position
        position_df["balance"] = position_df["free"] * position_df["price"]
        position_df["is_hold"] = position_df["balance"] > 5

        # 用当前价格，初始化已经持仓的成本价。"bid_price" -> cost price
        position_df["bid_price"] = 0.001
        condition = position_df["is_hold"]
        position_df.loc[condition, "bid_price"] = position_df.loc[condition, "price"]

        self.position_df = position_df
        self.account_value = self.position_df["balance"].sum()

        return
    
    def update_info(self, client):
        '''
        update free, locked value
        '''
        position_df = self.position_df
        
        # Get Latest account info
        account_infomation = client.account()  
        new_position_df = pd.DataFrame(account_infomation['balances'])

        # update USDT_value
        self.USDT_value = float(new_position_df[new_position_df["asset"] == "USDT"]['free'].values[0])

        # Drop and renew each latest positions
        position_df.drop(columns=["free", "locked", "price", "balance", "is_hold"], inplace=True)
        position_df.reset_index(inplace=True)

        position_df = pd.merge(position_df, new_position_df, on="asset", how='left')
        position_df.set_index('symbol', inplace=True, drop=True)

        # Update price data
        price_df = pd.DataFrame(client.ticker_price(symbols=list(position_df.index.values)))
        position_df = pd.merge(position_df, price_df, left_index=True, right_on='symbol', how='left')
        position_df.set_index('symbol', inplace=True, drop=True)

        # Formatting
        convert_dict = {'free': float,
                        'locked': float,
                        'price': float,
                        }
        
        position_df = position_df.astype(convert_dict)

        # Calculate if it is already in position
        position_df["balance"] = position_df["free"] * position_df["price"]
        position_df["is_hold"] = position_df["balance"] > 5

        #  Update account_value
        self.position_df = position_df
        self.account_value = self.position_df["balance"].sum()     
        return
    
    def get_symbols_held_sets(self):

        held_set = set(self.position_df[self.position_df["is_hold"]].index.values)
        un_held_set = set(self.position_df[~self.position_df["is_hold"]].index.values)
        return held_set, un_held_set
    
## Subclass 2
class Strategy_Info(Info_Interface):
    '''
    Strategy info
    '''
    def __init__(self, config_dict):

        # The super().__init__() line is calling the __init__ method of the parent class (Info_Interface) within the __init__ method of the Strategy_Info class.
        super().__init__()

        self.quote_currency = config_dict['quote_currency']
        self.candidate_symbols = config_dict['candidate_symbols']       
        self.base_currencies = [x[:-4] for x in config_dict['candidate_symbols']] # Remove 'USDT' from candidate symbols
        
        return
    
    def init_info(self, client):
        '''
        self.trade_info_df:
            dataframe
            index_col: symbol
            value_col: theta
        self.price_dict:
            {symbol: price_df}
        '''
        self.check_symbols(client)
        self.theta_info_df = pd.DataFrame({'symbol': self.candidate_symbols})  # theta_info_df
        self.theta_info_df['theta'] = 0.001
        self.theta_info_df.set_index('symbol', inplace=True, drop=True)
        self.get_trade_decimal_precision(client)  # 获取每个交易对支持的交易精度，加入到theta_info_df["stepDecimal"]

        self.price_dict = dict()  # price_dict
        return
    
    def check_symbols(self, client):
        '''
        检查候选的交易对是否存在于市场交易对象中
        '''
        self.candidate_symbols = [x.upper() for x in self.candidate_symbols]
        self.base_currencies = [x.upper() for x in self.base_currencies]

        exchange_dict = client.exchange_info()
        market_symbols = set([symbol_dict['symbol'] for symbol_dict in exchange_dict['symbols']])

        symbol_in_market = True

        for symbol in self.candidate_symbols:

            if symbol not in market_symbols:
                print(f"Warning: {symbol} is not in market_symbols")
                self.candidate_symbols.remove(symbol)
                symbol_in_market = False
                continue

        if symbol_in_market:
            print("All candidate symbols are in market_symbols")
        return
    
    
    def update_price_dict(self, price_dict):
        self.price_dict = price_dict
        return

    def get_trade_decimal_precision(self, client):

        self.theta_info_df["stepDecimal"] = 2

        for temp_symbol in self.candidate_symbols:

            print("temp_symbol:", temp_symbol)

            temp_info_dict = client.exchange_info(symbol=temp_symbol)
            temp_stepSize = temp_info_dict['symbols'][0]['filters'][1]['stepSize']
            temp_decimal_precision = get_decimal_precision(temp_stepSize)

            self.theta_info_df.loc[temp_symbol,"stepDecimal"] = temp_decimal_precision

        pass

class Info_Controller():
    '''
    info controller
    '''
    def __init__(self, config_dict, client, use_strategy=True):

        self.use_strategy = use_strategy

        self.account_info = Account_Info(config_dict)

        if self.use_strategy:
            self.strategy_info = Strategy_Info(config_dict)

        self._init_info_all(client)

    def _init_info_all(self, client):

        self.account_info.init_info(client)

        if self.use_strategy:
            self.strategy_info.init_info(client)

    def update_info_all(self, client):

        self.account_info.update_info(client)

        if self.use_strategy:
            self.strategy_info.update_info(client)


## Example
            
if __name__ == "__main__":

    util_config = dict(
        candidate_symbols=['AUTOUSDT', 'ETHBULLUSDT', 'BNBBULLUSDT', 'BNBBEARUSDT', 
        'SUSHIDOWNUSDT', 'TRBUSDT', 'AAVEUSDT', 'BTGUSDT', 'EOSBEARUSDT', 'BCHSVUSDT', 
        'LTCUSDT', 'ALCXUSDT', 'BEARUSDT', 'ETHBEARUSDT', 'COMPUSDT', 'EGLDUSDT', 
        'QNTUSDT', 'XRPBULLUSDT', 'FTTUSDT', 'RGTUSDT', 'ANYUSDT', 'SSVUSDT', 
        'GMXUSDT', 'HIFIUSDT', 'UNFIUSDT', 'EOSBULLUSDT', 'NMRUSDT', 'NANOUSDT', 
        'SOLUSDT', 'AUCTIONUSDT', 'MULTIUSDT', 'ATOMUSDT', 'XTZDOWNUSDT', 'ETCUSDT', 
        'AXSUSDT', 'USTUSDT', 'WLDUSDT', 'ARUSDT', 'CYBERUSDT', 'WINGUSDT', 'INJUSDT', 
        'RUNEUSDT', 'LINKUSDT', 'MTLUSDT',  'AVAXUSDT', 'XVSUSDT', 'FORTHUSDT', 
        'COCOSUSDT', 'OGUSDT'],
        quote_currency='usdt',
    )

    strategy_config = dict(
        market_t=100,
        fraction=0.3,  # 买入比例
    )
    
    config_dict = dict()
    config_dict.update(util_config)
    config_dict.update(strategy_config)

    API_urls = ["https://api.binance.com",
                "https://api-gcp.binance.com",
                "https://api1.binance.com",
                "https://api2.binance.com",
                "https://api3.binance.com",
                "https://api4.binance.com"]

    # API key/secret are required for user data endpoints
    client = SpotClient(api_key=g_api_key,
                        api_secret=g_secret_key,
                        base_url=API_urls[3])

    info_c = Info_Controller(config_dict, client)
    # print(info_c.account_info.position_df)
    # print(info_c.account_info.account_value)

    # symbol = "COMPUSDT"
    # info_c.account_info.position_df.loc[symbol, "free"] = 0.1

    # info_c.update_info_all(client)  # test update

    print(info_c.strategy_info.theta_info_df)