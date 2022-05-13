# -- coding: utf-8 --
import pandas as pd
import numpy as np
from alpha_Base import Alpha_Base
from copy import deepcopy as copy
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import h5py
from scipy import stats
import time
from common_lib import QLLIB
import matplotlib.pyplot as plt
import os
import sys


# reload(sys)
# sys.setdefaultencoding('utf-8')


class Alpha(Alpha_Base):

    def __init__(self, cf):
        Alpha_Base.__init__(self, cf)
        self.h5_path_5min = cf['h5_path_5min']
        self.h5_path_1min = cf['h5_path_1min']
        self.h5_path_tick = cf['h5_path_tick']

    #####################  ###################
    def demo_001(self):
        # 简单操作，依据两个信号直接合成板块信号，可用于计算板块动量等简单信号
        # 主要用于依据特征排序进行分类的场景
        # 主要函数: map_from_signal()
        # 市值作为分类风格、5日涨幅作为信号、分20档、板块内信号中位数作为板块信号、信号生成方式为map
        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        # level_num : 根据style_vect 分类的数量
        config = {}
        config['alpha_name'] = 'demo_001'
        config['alpha_num'] = 'demo_001'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        close_re = (self.close_p * self.re_p).rolling(window=20, min_periods=10).mean()

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            style_vect = self.cap_p.loc[di] * univers
            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers

            vect_alpha = QLLIB.map_from_signal(style_vect=style_vect, factor_vect=signal_vect, \
                                               retn_type='map', define_central='median', level_num=20) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_002(self):
        # 简单操作，依据分类信号将类别信息映射到每只股票，可用于计算板块动量等简单信号
        # 主要用于一些自定义的分类、行业分类等无法用特征排序的场景
        # 需要提前做好分类信息，输入的类别信息必须是哑变量矩阵，比如行业分类(分类信息一般每天一矩阵)
        # 主要函数: map_from_mtx()
        # 市值作为分类风格、5日涨幅作为信号、分20档、板块内信号中位数作为板块信号、信号生成方式为map
        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        config = {}
        config['alpha_name'] = 'demo_002'
        config['alpha_num'] = 'demo_002'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers

            vect_alpha = QLLIB.map_from_mtx(map_df=self.ind_dict[di], factor_vect=signal_vect, \
                                            retn_type='map', define_central='median') * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_003(self):
        # 复杂操作，根据分类风格和股票信号，先生成按天存储的股票分类映射字典（类似行业分类，每天有一个分类矩阵）
        # 和每个分类信号的时间序列，再基于该时间序列做复杂计算，比如计算均线、相关性、换手变化等
        # 最后，基于映射字典将计算出的复杂信号映射到每只股票上
        # 主要函数：generate_ts_map_info()  map_from_style()
        # 市值作为分类风格、计算每个类别涨幅的20日自相关性：先得到每个类别涨幅的时间序列->计算每个序列自相关性->映射回股票

        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        # level_num : 根据style_vect 分类的数量
        config = {}
        config['alpha_name'] = 'demo_003'
        config['alpha_num'] = 'demo_003'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        # 此处self.cap_p,self.chgRate_p都是系统自带数据，如果要用自己计算的风格因子和信号，
        # 需要先算出相应信号，再放到下面函数中。输入的两个dataframe的index 日期必须一致。
        map_dict, style_ts = QLLIB.generate_ts_map_info(self.cap_p, self.chgRate_p, define_central='median',
                                                        level_num=20)

        # 研究中可将中间结果保存到本地,重复利用，不必每次都重新生成。
        # 读文件的代码：result = QLLIB.read_from_selflib(data_name = 'close_p',data_path = '')
        # 上述代码要求被读入的文件必须有index和columns标签，如果没有，就自己写读文件的代码
        # style_ts.to_csv(……) ，
        # for i_dt in map_dict.keys():
        #    map_dict[i_dt].to_csv('……/%s.csv'%i_dt)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            map_df = map_dict[di]
            style_signal = QLLIB.self_corr_str(style_ts, style_ts.shift(1), axis_20, di)

            vect_alpha = QLLIB.map_from_style(map_df, style_signal) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_004(self):
        # 通用因子开发框架
        # 计算20日均线与60日均线的距离
        # 注：类对象默认只读取了begin_date 前40天的的数据，计算逻辑对数据的需求如果超过40天，要手动读入
        config = {}
        config['alpha_name'] = 'demo_004'
        config['alpha_num'] = 'demo_004'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        ########### 数据预处理：复权、均值、排序等向量化操作 #######
        close_p = QLLIB.read_from_selflib(data_name='close_p', data_path=self.data_path).reindex(columns=self.columns)
        re_p = QLLIB.read_from_selflib(data_name='re_p', data_path=self.data_path).reindex(columns=self.columns)

        MA_20 = (close_p * re_p).rolling(window=20, min_periods=8).mean()
        MA_60 = (close_p * re_p).rolling(window=60, min_periods=24).mean()
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            # axis_20  = self.trade_day[axis_now-20+1] #取前20个交易日日期

            vect_alpha = (MA_20.loc[di] / MA_60.loc[di] - 1.0) * univers

            # 因子标准化处理，一般此处不做标准化处理，后期会有统一处理
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_005(self):
        # 通用因子开发框架
        # 计算10日大单累计净流入
        # 资金流数据需要自己手动读入

        config = {}
        config['alpha_name'] = 'demo_005'
        config['alpha_num'] = 'demo_005'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        ########### 数据预处理：复权、均值、排序等向量化操作 #######
        BUY_VALUE_LARGE_ORDER = QLLIB.read_from_selflib(data_name='BUY_VALUE_LARGE_ORDER', data_path=self.data_path) \
            .reindex(columns=self.columns)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]  # 取前10个交易日日期

            vect_alpha = self_sum_str(BUY_VALUE_LARGE_ORDER, axis_10, di) * univers

            # 因子标准化处理，一般不做标准化处理
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_006(self):
        # 通用因子开发框架
        # 计算10日价量相关性
        # 直接使用对象已有的成员数据，调用计算相关系数的函数self_corr_str
        config = {}
        config['alpha_name'] = 'demo_006'
        config['alpha_num'] = 'demo_006'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        ########### 数据预处理：复权、均值、排序等向量化操作 #######
        close_temp = self.close_p * self.re_p
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]  # 取前10个交易日日期

            vect_alpha = QLLIB.self_corr_str(close_temp, self.volume_p, axis_10, di) * univers

            # 因子标准化处理，一般不做标准化处理
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config


def ind_rank(df_source, ind_ref):
    # 行业内rank
    ind_source = ind_ref * df_source
    rank_result = ind_source.rank(axis=1)
    max_value = rank_result.max(axis=1)
    result = ((rank_result.T) / max_value).T
    return result.sum()


def self_mean_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.mean(axis=0)


def self_sum_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.sum(axis=0)


def self_kurt_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.kurtosis(axis=0)


def self_skew_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.skew(axis=0)


def self_tsrank_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.rank().loc[now]


def self_max_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.max(axis=0)


def self_idxmax_str_np(df_source, last, now):
    mtx = df_source.loc[last:now]
    vect = np.argsort(-mtx.values, axis=0)[0, :]
    return pd.Series(vect, index=df_source.columns)


def self_idxmax_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmax(axis=0)


def self_idxmin_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmin(axis=0)


def z_score_org(vect):
    return (vect - vect.mean()) / vect.std()


def self_min_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.min(axis=0)


def self_std_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.std(axis=0)


def self_sigmoid(vect):
    result = 1 / (1 + np.exp(-vect))
    return result


def self_scale(df_source):
    return df_source / df_source.sum()


def MAD_Outlier(arr):
    arr = arr.astype(float)
    if sum(np.isnan(arr.astype(float))) == len(arr):
        return arr
    median = np.nanmedian(arr)
    MAD = np.nanmedian(np.abs(arr - median))
    arr[arr > median + 6 * 1.4826 * MAD] = median + 6 * 1.4826 * MAD
    arr[arr < median - 6 * 1.4826 * MAD] = median - 6 * 1.4826 * MAD
    return arr


def std_vect_org(vect):
    result = (vect - vect.mean()) / vect.std()
    return result


def std_vect(vect, level=10):
    med = vect.median()
    err = (vect - med).abs().median()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


def std_vect_mtx(df_source, level=6):
    result = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    for di in df_source.index:
        vect = df_source.loc[di]
        vect = std_vect(vect)
        m = result.loc[di]
        m[:] = vect
    return result


def std_vect_mean(vect, level=20):
    med = vect.mean()
    err = (vect - med).abs().std()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


def self_rank_mtx(df_source):
    df_rank = df_source.rank(axis=1)
    max_value = df_rank.max(axis=1)
    df_rank = ((df_rank.T) / max_value).T
    return df_rank


def self_EMA(df_source, last, now):
    new_source = df_source.loc[last:now]
    length = new_source.shape[0]

    weight = copy(new_source)
    weight = weight * 0.0 + 1

    m = pd.Series(index=new_source.index)

    ratio = pd.Series(range(length + 1)[1:])

    m[:] = ratio
    new_source = (new_source.T * m).T

    weight = ((weight.T) * m).T

    new_source = new_source.sum() / weight.sum()
    return new_source


def self_decay_str(df_source, last, now, length):
    ratio = pd.Series(range(length + 1)[1:])
    new_source = (copy(df_source.loc[last:now]).T)
    m = pd.Series(index=new_source.columns)
    m[:] = ratio
    new_source = new_source * m
    new_source = new_source.sum(axis=1) / (length * (length + 1) * 0.5)
    return new_source.T


def std_solve(df_source, length=20):
    mean_source = copy(df_source)
    std_source = copy(df_source)
    days = len(df_source.index)
    for i in range(days):
        begin = i - length
        if begin < 0:
            begin = 0
        m = mean_source.iloc[i]
        m[:] = df_source.iloc[begin:i].mean()
        n = std_source.iloc[i]
        n[:] = df_source.iloc[begin:i].std()
    return (df_source - mean_source) / std_source


def self_normalize(df_source):
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.0000001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


def rank(df_source):
    # 与self_rank
    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)

    #


def ind_neutral(df_source, ind_p):
    result = ind_p * df_source

    mean = result.mean(axis=1)
    std = result.std(axis=1) + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


def self_free_neutral(df_source, objt):
    result = objt * df_source
    mean = result.mean(axis=1)
    std = result.std(axis=1) + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


def GetZScoreFactor(factor_df):
    ## use zscore x-mean/std
    zscore_factor_df = factor_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1, raw=False)
    return zscore_factor_df


def self_get_objt(df_source):
    # df_source:
    rank1 = rank(df_source)
    result = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=df_source.index)
    for i in range(10):
        m = copy(rank1)
        up = 1.0 - 0.1 * i
        down = up - 0.1
        m[m > up] = np.nan
        m[m < down] = np.nan
        m[m > 0.0] = 1.0

        result.iloc[i][:] = m
    return result


def ema(arr):
    window = len(arr)
    weight = [1.0 / (window - idx) for idx in range(0, window, 1)]
    weight = np.where(np.isnan(arr), 0.0, weight)
    weight_sum = np.sum(weight)
    weight = weight / weight_sum
    return np.nansum(arr * weight)


def ema_psx(arr):
    window = len(arr)
    weight = [idx + 1 for idx in range(0, window, 1)]
    weight = np.where(np.isnan(arr), 0.0, weight)
    weight_sum = np.sum(weight)
    weight = weight / weight_sum
    return np.nansum(arr * weight)













































