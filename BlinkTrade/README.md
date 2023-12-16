# BlinkTrade
> A project for algorithmic trading of cryptocurrencies.

# Introduction 
实现很稳定的盈利！


# Strcuture
Does such a simple project really need an introduction to the file structure?
- [Bot](#Bot) - Relevant code and information about trading strategies and the bot are still being written...（交易策略和bot的相关代码，相关介绍还在撰写中……）
  - models - 模型
  - strategies - 策略相关代码
    - reverse_detector.py - 调用ml，判断反转点
    - strategies.py - 策略本身的代码
    - strategy_utils.py
  - utils
    - data_utils.py
    - information.py  - 管理策略运行时需要的全部数据。暂存数据。
    - trade_utils.py  - Utility functions for placing orders.(下单相关的工具函数)
  - run_trade.py - Main program to run the overall trading bot, calling various trading strategy classes. (运行整体交易bot，调用各类交易策略的主程序)


## Next steps
近期会尽量完善注释，更新相关research代码。（We will strive to improve comments and update the relevant research code in the near future.）
