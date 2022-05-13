# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:59:11 2021


"""

import pandas as pd
import numpy as np

from utils import BacktestUtils
from utils import PerfUtils

class IndustryEvent:

    def __init__(self):
    
        # 读取交易日数据
        self.daily_dates = pd.read_pickle("data/basic/daily_dates")
        
        # 读取股票收盘价数据
        self.stock_close = pd.read_pickle("data/daily/stock/S_DQ_ADJCLOSE")
        
        # 读取调研次数指标
        self.df_survey = pd.read_pickle("data/daily/other/stock_survey")
        
        # 读取股票换手率
        self.turn = pd.read_pickle("data/daily/stock/S_DQ_TURN")
                
        # 读取基准指数收盘价数据
        self.index_close = pd.read_pickle("data/daily/index/index_close")
        
        # 读取股票预处理相关数据
        self.updown = pd.read_pickle("data/daily/stock/UP_DOWN_LIMIT_STATUS")
        self.trade_st = pd.read_pickle("data/daily/stock/S_DQ_TRADESTATUS")
        self.st_mark = pd.read_pickle("data/daily/stock/ST_mark")
        self.listed_days  = pd.read_pickle("data/daily/stock/listed_days")
        
        # 过滤器：滤除涨跌停、换手率为0、非交易状态、ST状态、上市天数小于180的股票
        # 股票持有时间比较长，不能剔除涨跌停
        self.stock_filter = (self.updown.abs() != 1) & (self.turn > 1e-8) & \
                            (self.trade_st == "交易") & (self.st_mark != 1) & \
                            (self.listed_days >= pd.Timedelta("180 days"))
                            
                            
    # =========================================================================
    # 生成调仓信号           
    # =========================================================================
    def portfolio_gen(self, lookback_days, holding_days, signal_number, stock_pool=None):
                        
        # 调研信号
        survey_sum = self.df_survey.rolling(lookback_days).sum()
        
        # 填充空值
        survey_sum[survey_sum.isnull()] = 0
                
        # 将数据向后平移，即认为机构调研数据有一定的滞后性
        survey_sum_rolling = survey_sum.rolling(holding_days).max()
                     
        # 按照之前的交易日序列计算汇总信号数据时对应的最后一个交易日
        signal_freq = survey_sum_rolling.loc[self.daily_dates.loc[
            survey_sum_rolling.index[0]:survey_sum_rolling.index[-1]]]
                                        
        # 当行业指标给定的最后日期大于最后一个交易日时, 踢除无法调仓的最后交易信号（月末容易出现）
        if signal_freq.index[-1] >= self.daily_dates[-1]:
            signal_freq = signal_freq.drop(index=signal_freq.index[-1])
            
        # 调仓日期为生成信号的下一天，即月初第一个交易日               
        signal_freq.index = [self.daily_dates.iloc[
            self.daily_dates.index.tolist().index(i) + 1] for i in signal_freq.index]
        
        # 生成具体持仓信息
        signal_adj = signal_freq.replace(0, np.nan) >= signal_number
            
        # 信号过滤，剔除异常股票
        # 如果上一个持仓日持有该股票，不需要卖出，只对买入股票进行判断
        signal_adj[(~self.stock_filter.loc[signal_adj.index, signal_adj.columns]) &
                   (signal_adj.shift(1) == False)] = False
        
        # 生成仓位矩阵，没有持仓信息时用零值代替
        long_port = signal_adj.div(signal_adj.sum(axis=1), axis=0).fillna(0)
        
        # 填充空值
        long_port[long_port.isnull()] = 0
        
        # 日期调整
        return long_port


    # =============================================================================
    #  回测主程序
    # =============================================================================
    def backtest(self, arg):
        
        # 生成调仓信号
        long_port_all = self.portfolio_gen(
            lookback_days=arg['回看天数'], holding_days=arg['持仓天数'], 
            signal_number=arg['机构调研数量'])
        
        # 按日期截断
        long_port = long_port_all.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 提取沪深300指数作为基准
        base_nav = self.index_close.loc[arg['回测开始时间']:, '000985.CSI']
            
        # 回测程序
        [nav, turn] = BacktestUtils.cal_nav(long_port, self.stock_close, base_nav, arg['手续费'])
        
        # 起始点调整
        base_nav = base_nav.reindex(nav.index)
        base_nav /= base_nav[0]
        
        # 净值拼接
        nav = pd.concat([nav, base_nav], axis=1)
        nav.columns = ['策略', '基准']
                    
        # 计算月度调仓信号，用于计算胜率
        nav_resample = nav.resample('M').last()
        monthly_dates = self.daily_dates.resample('M').last()
        panel_dates = pd.to_datetime(monthly_dates[nav_resample.index].values)
        
        # 计算策略表现
        perf = PerfUtils.excess_statis(nav['策略'], nav['基准'], panel_dates)
           
        # 给出序号的时候进行存储操作
        if '序号' in arg.keys():
            print(arg['序号'], perf.loc['策略', '年化超额收益率'])
            # nav.to_pickle('results/事件驱动测试/遍历测试净值/{}'.format(arg['序号']))
            perf.to_pickle('results/事件驱动测试/遍历测试风险收益指标/{}'.format(arg['序号']))
            
        # 否则直接返回结果即可
        else:
            return nav, long_port, perf, turn
       

if __name__ == "__main__":
            
    # 模型初始化
    model = IndustryEvent()
                
    # 测试
    nav, long_port, perf, turn  = model.backtest({
            '回看天数': 60, '持仓天数': 100, 
            '机构调研数量': 50, '回测开始时间': '2015-01-01',
            '回测结束时间': '2021-02-28',  '手续费': 0.001})

