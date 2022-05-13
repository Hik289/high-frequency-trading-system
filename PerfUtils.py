# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:59:50 2018

@author: licong
"""

import pandas as pd
import numpy as np


# # -----------------------------------------------------------------------------
# # 入口函数一：统计净值曲线的基本业绩指标：年化收益率、年化波动率、夏普比率、最大回撤
# # [输入]
# # df_nav   净值曲线，Series或DataFrame，每一列代表一个策略的净值走势
# # -----------------------------------------------------------------------------
# def normal_statis(df_nav):
    
#     # 获取所有净值曲线的年化收益率与年化波动率
#     perf = df_nav.apply([annualized_return, sharp_ratio, max_drawdown])
    
#     # 设置指标名称
#     perf.index = ['超额年化收益', '多空信息比', '多空最大回撤']

#     return perf.T


# -----------------------------------------------------------------------------
# 入口函数一：统计净值曲线的基本业绩指标：年化收益率、年化波动率、夏普比率、最大回撤
# [输入]
# df_nav   净值曲线，Series或DataFrame，每一列代表一个策略的净值走势
# -----------------------------------------------------------------------------
def normal_statis(df_nav):
    
    # 获取所有净值曲线的年化收益率与年化波动率
    perf = df_nav.apply([annualized_return, annualized_volatility, sharp_ratio, max_drawdown])
    
    # 设置指标名称
    perf.index = ['年化收益率', '年化波动率', '夏普比率', '最大回撤']

    return perf.T


# -----------------------------------------------------------------------------
# 入口函数二：统计单个策略净值曲线的基本指标，以及相对于基准的超额指标
# [输入]
# nav_seq        策略净值
# base_seq       基准净值
# refresh_dates  调仓日期
# -----------------------------------------------------------------------------
def excess_statis(nav_seq, base_seq, refresh_dates):
    
    # 初始化结果矩阵
    perf = pd.DataFrame(index=['策略', '基准'], 
                        columns=['年化收益率','年化波动率','夏普比率','最大回撤',
                                 '年化超额收益率','超额收益年化波动率','信息比率',
                                 '超额收益最大回撤','调仓胜率', '相对基准盈亏比'])
    
    # 统计策略和基准的基本业绩指标
    perf.iloc[:, :4] = normal_statis(pd.concat([nav_seq, base_seq], axis=1)).values
    
    # 计算策略相比于基准的超额收益
    excess_return = nav_seq.pct_change() - base_seq.pct_change()
    
    # 计算超额收益率累计净值
    excess_nav = (1 + excess_return.fillna(0)).cumprod()
    
    # 计算超额收益的业绩指标
    perf.iloc[0, 4:8] = normal_statis(excess_nav).values
    
    # 计算胜率
    perf.loc['策略', '调仓胜率'] = win_rate(nav_seq, base_seq, refresh_dates)

    # 计算胜率
    perf.loc['策略', '相对基准盈亏比'] = profit_loss_ratio(nav_seq, base_seq, refresh_dates)
    
    return perf


# -----------------------------------------------------------------------------
# 年化收益率
# -----------------------------------------------------------------------------
def annualized_return(nav):
    
    return pow(nav[-1] / nav[0], 250/len(nav)) - 1


# -----------------------------------------------------------------------------
# 年化波动率
# -----------------------------------------------------------------------------
def annualized_volatility(nav):

    return nav.pct_change().std() * np.sqrt(250)


# -----------------------------------------------------------------------------
# 夏普比率
# -----------------------------------------------------------------------------
def sharp_ratio(nav):

    return annualized_return(nav) / annualized_volatility(nav)
        

# -----------------------------------------------------------------------------
# 最大回撤
# -----------------------------------------------------------------------------  
def max_drawdown(nav):
    
    # 初始化
    max_drawdown = 0
    
    # 遍历之后每一天
    for index in range(1, len(nav)):
        
        cur_drawdown = nav[index] / max(nav[0:index]) - 1
        
        if cur_drawdown < max_drawdown:
            max_drawdown = cur_drawdown
            
    return max_drawdown


# -----------------------------------------------------------------------------
# 调仓胜率
# -----------------------------------------------------------------------------  
def win_rate(strategy_nav, base_nav, refresh_dates):
    
    # 抽取调仓日策略和基准的净值
    resampled_strategy = strategy_nav[refresh_dates].tolist()
    resampled_base = base_nav[refresh_dates].tolist()
    
    # 如果调仓日最后一天和净值序列最后一天不一致，则在尾部添加最新净值
    if strategy_nav.index[-1] != refresh_dates[-1]:
        resampled_strategy.append(strategy_nav[-1])
        resampled_base.append(base_nav[-1])

    # 计算调仓超额收益
    resampled_strategy = pd.Series(resampled_strategy)
    resampled_base = pd.Series(resampled_base)
    excess = resampled_strategy.pct_change().dropna() - resampled_base.pct_change().dropna()
    
    return (excess > 0).sum() / len(excess)
    

# -----------------------------------------------------------------------------
# 调仓胜率
# -----------------------------------------------------------------------------  
def profit_loss_ratio(strategy_nav, base_nav, refresh_dates):
    
    # 抽取调仓日策略和基准的净值
    resampled_strategy = strategy_nav[refresh_dates].tolist()
    resampled_base = base_nav[refresh_dates].tolist()
    
    # 如果调仓日最后一天和净值序列最后一天不一致，则在尾部添加最新净值
    if strategy_nav.index[-1] != refresh_dates[-1]:
        resampled_strategy.append(strategy_nav[-1])
        resampled_base.append(base_nav[-1])

    # 计算调仓超额收益
    resampled_strategy = pd.Series(resampled_strategy)
    resampled_base = pd.Series(resampled_base)
    excess = resampled_strategy.pct_change().dropna() - resampled_base.pct_change().dropna()
    
    return - excess[excess > 0].mean() / excess[excess < 0].mean()
    
