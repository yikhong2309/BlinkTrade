import pandas as pd

def calculate_position_value(price, amount):
    '''计算base currency持仓的价值'''
    return price * amount

def np_round_floor(number, decimals):  # todo 确认对负数是否适用
    multiplier = 10 ** decimals
    return int(number * multiplier) / multiplier

def pandas_fill(arr):
    df = pd.DataFrame(arr)
    df.fillna(0, axis=0, inplace=True)
    out = df[0].values
    return out


origin_cols = [  # 8个 +  'volume_obv' 无法直接使用
    'trend_adx', 'trend_cci', 'macd',
    'momentum_rsi',  'volume_vwap',
    'volatility_atr', 'bop', 'ohlc4'
]
Z_cols = [f'{col_name}_Z' for col_name in origin_cols]
Z_diff1_cols = [f'{col_name}_diff_1' for col_name in Z_cols]
Z_diff2_cols = [f'{col_name}_diff_2' for col_name in Z_cols]

# 单个币种的全部特征
X_standalone_cols = Z_cols + Z_diff1_cols + Z_diff2_cols  # 27个

# 交互特征  18*2 = 36个
co_diff_target_cols = Z_cols + Z_diff1_cols # 对于全部Z指标及其一阶差分指标都做跨币种差分
co_diff_target_cols_btc = ['co_diff_'+x+"_btc" for x in co_diff_target_cols]
co_diff_target_cols_eth = ['co_diff_'+x+"_eth" for x in co_diff_target_cols]

# btc eth自身的特征  27*2 = 54
btc_X_cols = [x+"_btc" for x in X_standalone_cols]
eth_X_cols = [x+"_eth" for x in X_standalone_cols]

# 所有X_cols
all_X_cols = origin_cols + X_standalone_cols + co_diff_target_cols_btc + co_diff_target_cols_eth + btc_X_cols + eth_X_cols
useful_X_Cols = X_standalone_cols + co_diff_target_cols_btc + co_diff_target_cols_eth + origin_cols ## 目前来看更有效的特征子集

# 所有y_cols
y_cols = ['PE_1_real', 'PE_1']
for i in [5,10,20]:
    y_cols.append('PE_'+str(i)+'_mean')
    y_cols.append('PE_'+str(i)+'_real')