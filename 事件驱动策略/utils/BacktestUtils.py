# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:56:20 2018

"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# 计算策略净值曲线
# df_port         调仓矩阵，行索引为调仓日期，列索引为所有股票
#                 每一行内容对应当前调仓所有股票对应的权重，总和为1
# backtest_close  回测区间所有股票的收盘价
# fee             单边调仓手续费
# -------------------------------------------------------------------------    
def cal_nav(df_port, backtest_close, base_nav, fee):        
        
    # 获取所有调仓日期
    refresh_dates = df_port.index.tolist()
    
    # 节选出回测区间内的收盘价
    backtest_close = backtest_close.loc[refresh_dates[0]:,:]
    
    # 获取回测区间日频交易日
    backtest_dates = backtest_close.index.tolist()
            
    # 初始化净值曲线
    nav = pd.Series(index=backtest_dates, name='策略', dtype=float)
    
    # 初始化换手率记录
    turn = pd.Series(index=refresh_dates, name='当期换手', dtype=float)
    
    # 初始化日期计数器
    date_index = 0  
    
    # # 实际的调仓日期，保存下来之后备用
    # actual_change_date = list()
    
    # 遍历每个日期
    for date_index in range(len(backtest_dates)):
        
        # -----------------------------------------------------------------
        # 获取对应日期
        # -----------------------------------------------------------------
        date = backtest_dates[date_index]
        
        # -----------------------------------------------------------------
        # 如果是回测期首日，则执行初次建仓
        # -----------------------------------------------------------------
        if date_index == 0:  
            
            # 获取当前调仓权重
            new_weight = df_port.loc[date,:] 
            
            # # 第一个调仓日期
            # actual_change_date.append(date)
            
            # 计算当前持仓个股净值，考虑第一次调仓的手续费
            portfolio = (1 - fee) * new_weight
            
            # 记录净值
            nav[date] = 1 - fee
            
            # 跳过后面的代码，直接进行下一次循环
            continue
        
        # -----------------------------------------------------------------
        # 每到一个日期，都根据个股涨跌幅更新组合净值，将日期计数器自增1
        # -----------------------------------------------------------------
        # 当天收盘价
        cur_close = backtest_close.iloc[date_index, :]
        
        # 上一天的收盘价
        prev_close = backtest_close.iloc[date_index-1, :]

        # 判断最新的收盘价是否存在空值
        cur_close_nan = cur_close[cur_close.isna()].index
        
        # 当存在持有资产价格为空的情况时，重新计算权重分布，剔除此种资产
        # 此种情况很少见，不做细节处理
        if np.nansum(portfolio[cur_close_nan])> 0:
            
            # 提取前一个日期
            prev_date = backtest_dates[date_index-1]
            
            # 归一化当前持仓中个股权重, 空值记为0
            old_weight = portfolio / np.nansum(np.abs(portfolio))
            old_weight[old_weight.isnull()] = 0
            
            # 获取最新的持仓权重
            new_weight = old_weight.copy()
            new_weight[cur_close_nan]=0

            # 归一化当前持仓中个股权重, 空值记为0
            new_weight = new_weight / np.nansum(np.abs(new_weight))
            new_weight[new_weight.isnull()] = 0
            
            # 直接按照新的持仓组合分配权重
            portfolio = new_weight * nav[prev_date]
            
        # 根据涨跌幅更新组合净值
        portfolio = cur_close / prev_close * portfolio

        # 未持有资产时，组合净值维持不变
        if np.nansum(portfolio) == 0:
            nav[date] = nav.iloc[backtest_dates.index(date) - 1]
        else:
            nav[date] = np.nansum(portfolio)
            
        # -----------------------------------------------------------------
        # 如果当前是调仓日，还需执行调仓操作
        # -----------------------------------------------------------------
        if date in refresh_dates:

            # # 保存换仓日期
            # actual_change_date.append(backtest_dates[date_index])
                            
            # 归一化当前持仓中个股权重
            old_weight = portfolio / np.nansum(np.abs(portfolio))
            old_weight[old_weight.isnull()] = 0
            
            # 获取最新的持仓权重
            new_weight = df_port.loc[date,:] 
                        
            # 计算换手率，最小为0，也即不换仓，最大为2，也就是全部换仓
            turn_over = np.sum(np.abs(new_weight - old_weight))
            turn[date] = turn_over / 2
            
            # 更新换仓后的净值，也即扣除手续费
            nav[date] = nav[date] * (1 - turn_over * fee)
            
            # 更新持仓组合中个股的最新净值
            portfolio = new_weight * nav[date]
            
    
    # # 导出调仓日期
    # actual_change_date = pd.DataFrame(actual_change_date, columns=['date'])
    
    return nav, turn

