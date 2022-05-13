# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)

# =============================================================================
# 拥挤度指标计算主程序
# =============================================================================
class crowd_factors():
    
    def __init__(self):

        
        # 读取行业指数列表
        industry_list = pd.read_pickle('./data/indus_info')['行业名称'].tolist()
        self.industry_list = [i for i in industry_list if i not in ["综合金融","多元金融","保险Ⅱ"]]
        
        # 个股收盘价数据
        self.stock_close = pd.read_pickle("./data/stock_close")
        self.stock_daily_ret = self.stock_close/self.stock_close.shift(1)-1
        
        # 指数收盘价数据
        self.index_close = pd.read_pickle("./data/Wind_indus_close").resample('D').last()
        self.index_close = self.index_close.reindex(columns=self.industry_list)
        self.index_daily_ret = self.index_close.pct_change()
        index_excess = (self.index_daily_ret.T - self.index_daily_ret.mean(axis = 1)).T + 1
                
        self.index_daily_net_ret = index_excess.cumprod(skipna = True)
        self.index_daily_net_ret[np.isnan(self.index_daily_ret)] == np.nan
        
        # 涨跌停数据
        self.updown_limit = pd.read_pickle("./data/updown_limit")
        self.updown_limit = self.updown_limit.reindex(index=self.index_close.index)

        # 股票换手率数据
        self.stock_turn = pd.read_pickle("./data/stock_turn")
        self.stock_turn = self.stock_turn.reindex(index=self.index_close.index)
        
        # 中信行业归属数据
        self.indus_belong = pd.read_pickle("./data/indus_belong")
        self.indus_belong = self.indus_belong.reindex(index=self.index_close.index)
        
        # 读取数据有效性判断
#         self.basic_cond = pd.read_pickle("./data/daily/stock/basic_cond")
#         self.stock_close[~self.basic_cond] = np.nan
#         self.stock_daily_ret[~self.basic_cond] = np.nan
#         self.updown_limit[~self.basic_cond] = np.nan
#         self.indus_belong[~self.basic_cond] = np.nan
        
        # 行业指数数据：成交量、成交额、换手率
        self.index_volume = pd.read_pickle("./data/index_volume")
        self.index_amount = pd.read_pickle("./data/index_amount")
        self.index_turn = pd.read_pickle("./data/index_turn")

        行业成份股数量
        self.comp_num = pd.DataFrame()
        for idx in self.industry_list:
            self.comp_num[idx] = self.indus_belong[self.indus_belong == idx].count(axis=1)
        self.comp_num = self.comp_num[self.comp_num>=5]

    
    # 指标生成：行业动量指标
    # factor_name是指标名称，window是区间长度
    def gen_indus_momentum(self, factor_name, window):
        
        # window日收益率
        #momentum = self.index_daily_ret.rolling(window).mean()
        momentum = self.index_daily_net_ret.rolling(window).mean()
        # 普通动量
        if factor_name == "normal_momentum":
            factor = momentum
        
        # 路径调整动量
        if factor_name == "distance_momentum":
            
            # 每日收益率的绝对值
            daily_retabs = self.index_daily_ret.abs()
            #daily_retabs = self.index_daily_net_ret.abs()            
            # 区间求和计算绝对长度
            rolling_sum = daily_retabs.rolling(window).sum()
            
            factor = self.index_close.pct_change(window) / rolling_sum
            
        # 夏普比率
        if factor_name == "sharpe_momentum":
            
            # 计算区间平均收益率
            #rolling_mean = self.index_daily_ret.rolling(window).mean()
            rolling_mean = self.index_daily_net_ret.rolling(window).mean()            
            # 区间收益率的标准差
            #rolling_std = self.index_daily_ret.rolling(window).std()
            rolling_std = self.index_daily_net_ret.rolling(window).std()            
            factor = rolling_mean/rolling_std
            
        # 信息比率
        if factor_name == "information_momentum":
            
            # 行业等权日收益率
            #daily_market_return = self.index_daily_ret.mean(axis=1)
            daily_market_return = self.index_daily_net_ret.mean(axis=1)            
            # 超额收益
            #excess_return = self.index_daily_ret.sub(daily_market_return,axis=0)
            excess_return = self.index_daily_net_ret.sub(daily_market_return,axis=0)            
            # 超额收益的平均值和标准差
            ex_rolling_mean = excess_return.rolling(window).mean()
            ex_rolling_std = excess_return.rolling(window).std()
            
            factor = ex_rolling_mean/ex_rolling_std
        
        # 尾部动量
        if factor_name == "tail_momentum":
            
            # 短期窗口期内的1%VaR
            #factor = self.index_daily_ret.rolling(window).quantile(0.01)
            factor = self.index_daily_net_ret.rolling(window).quantile(0.01)        
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：流动性
    def gen_indus_flow_factor(self,factor_name,window):
        
        # 读取数据
        if factor_name == "volume":
            data = self.index_volume
        if factor_name == "amount":
            data = self.index_amount
        if factor_name == "turn":
            data = self.index_turn
        
        # 计算滚动平均值
        factor = data.rolling(window).mean()
        
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：乖离率
    def gen_indus_bias_factor(self,factor_name,window):

        # 读取数据               
        if factor_name == "close_bias":
            df = self.index_close
        if factor_name == "volume_bias":
            df = self.index_volume
        if factor_name == "amount_bias":
            df = self.index_amount
        if factor_name == "turn_bias":
            df = self.index_turn
                
        # 计算滚动平均值
        rolling_mean = df.rolling(window).mean()
        
        # 计算数据偏离度
        factor = (df-rolling_mean)/rolling_mean.abs()
            
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：滚动相关系数
    def gen_roll_corr_factor(self,factor_name,window):
        
        # 读取数据
        if factor_name == "corr_volume_close":
            df = self.index_volume
        if factor_name == "corr_amount_close":
            df = self.index_amount
        if factor_name == "corr_turn_close":
            df = self.index_turn
        
        factor = pd.DataFrame()
        
        # 计算相关系数
        factor = df.rolling(window).corr(self.index_close)
                       
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：波动率
    def gen_indus_vol_factor(self,factor_name,window):

        # 日收益率
        #indus_return = self.index_daily_ret
        indus_return = self.index_daily_net_ret        
        if factor_name == 'vol':            
            factor = indus_return.rolling(window).std()
            
        if factor_name == 'downvol':    
            
            # 求取下行收益率
            index_down = indus_return < 0
            down_indus_return = indus_return[index_down]
            down_indus_return[down_indus_return.isnull()] = 0   
               
            factor = down_indus_return.rolling(window).std()

        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：分布特征
    def gen_indus_distribution_factor(self,factor_name,window):
        
        # 求取收益率
        daily_ret = self.index_daily_ret
        
        # 偏度
        if factor_name == 'skewness':            
            factor = daily_ret.rolling(window).skew()
                        
        # 峰度
        elif factor_name == 'kurtosis':            
            factor = daily_ret.rolling(window).kurt()
                        
        # 协偏度
        elif factor_name == 'coskewness':
            
            # 滚动均值
            daily_ret_mean = daily_ret.rolling(window).mean()
            daily_excess_ret = daily_ret-daily_ret_mean
            
            # 基准收益率
            base_return = daily_ret.mean(axis=1)
            rolling_base_mean = base_return.rolling(window).mean()
            base_excess_ret = base_return-rolling_base_mean
            
            # 计算分子
            numerator_origin = daily_excess_ret.mul(base_excess_ret**2,axis=0)
            numerator = numerator_origin.rolling(window).sum()
            
            # 计算分母
            denominator_origin = base_excess_ret**3
            denominator = denominator_origin.rolling(window).sum()
            
            factor = numerator.div(denominator,axis=0)
        
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # 指标生成：涨跌比例
    # factor_name选择指标类型，para补充上涨/下跌的阈值，comp_num所有行业的成份股数量
    def gen_updown_factor(self, factor_name, window, para_pou=0):
                
        # 设置附加条件
        # 行业内股票涨幅超过para的比例
        if (factor_name == "up_pct") & (para_pou != None):
            
            # 对水平方向统计收益率 >= para的个数
            spec_cond = self.stock_daily_ret >= para_pou
            
        # 行业内股票跌幅超过para的比例
        elif (factor_name == "down_pct") & (para_pou != None):
            
            # 对水平方向统计收益率 <= para的个数
            spec_cond = self.stock_daily_ret >= para_pou
            
        # 行业内股票涨停比例
        elif factor_name == "limitup_pct":
            
            # 对水平方向统计涨停股票的个数
            spec_cond = self.updown_limit == 1
            
        # 行业内股票跌停比例
        elif factor_name == "limitdown_pct":
            
            # 对水平方向统计跌停股票的个数
            spec_cond = self.updown_limit == -1
        
        factor = pd.DataFrame()
        
        # 按行业循环
        for idx in self.industry_list:
            factor[idx] = (spec_cond & (self.indus_belong == idx)).sum(axis=1)

        # 上涨比例 = 上涨个数/成份股个数
        factor /= self.comp_num
        
        # 过去N天均值
        factor = factor.rolling(window).mean()

        # 计算分位数，调整方向
        quantile = self.gen_comp_factor(factor, factor_name,"rolling")
        
        return quantile

    
    # 指标生成：涨跌分布特征
    # factor_name选择标准差、偏度、峰度
    def gen_comp_distribution_factor(self, factor_name, window):

        factor = pd.DataFrame()
        
        # 收益率计算
        stock_ret = self.stock_close / self.stock_close.shift(window) - 1
#         stock_ret[~self.basic_cond] = np.nan
        
        # 换手率计算
        stock_turn = self.stock_turn.rolling(window).mean()
#         stock_turn[~self.basic_cond] = np.nan
        
        # 按行业循环
        for idx in self.industry_list:
            
            if factor_name == "comp_ret_vol":
                factor[idx] = stock_ret[self.indus_belong == idx].std(axis=1)
                
            if factor_name == "comp_ret_skewness":
                factor[idx] = stock_ret[self.indus_belong == idx].skew(axis=1)
                
            if factor_name == "comp_ret_kurtosis":
                factor[idx] = stock_ret[self.indus_belong == idx].kurt(axis=1)

            if factor_name == "comp_turn_vol":
                factor[idx] = stock_turn[self.indus_belong == idx].std(axis=1)
                
            if factor_name == "comp_turn_skewness":
                factor[idx] = stock_turn[self.indus_belong == idx].skew(axis=1)
                
            if factor_name == "comp_turn_kurtosis":
                factor[idx] = stock_turn[self.indus_belong == idx].kurt(axis=1)
                
        # 成份股数量 >= 5
        factor = factor[self.comp_num.notnull()]

        # 计算分位数，调整方向
        quantile = self.gen_comp_factor(factor, factor_name, "rolling")
        
        return quantile
      
    
    # 计算行业指标分位数，转换为拥挤度指标
    def gen_normal_factor(self,factor,factor_name,method,window = 250*5):
            
        factor = factor.dropna(how="all")
        
        rank_apply = lambda x: np.searchsorted(x,x[-1],sorter=np.argsort(x))/(len(x)-1)
        
        if method == "expanding":
            factor_quantile = factor.expanding(1).apply(rank_apply)
        if method == "rolling":
            factor_quantile = factor.rolling(window).apply(rank_apply)        
        
        # 调整方向
        inv_list = ["corr_volume_close","corr_amount_close","corr_turn_close",
                    "skewness","coskewness"]
        
        if factor_name in inv_list:
            factor_quantile = 1-factor_quantile
        
        factor_quantile = updown_restriction(updown = 0,factors = factor_quantile)
        
        return factor_quantile
    
    
    # 计算成份股指标的分位数，转换为拥挤度指标
    def gen_comp_factor(self,factor,factor_name,method,window = 250*5):
        
        # 初始化输出结果
        output = factor.copy()*np.nan
        
        # 剔除银行证券行业（因个股数较少出现过缺失现象）
        new_industry_list = [i for i in self.industry_list if i not in ["银行","证券Ⅱ"]]

        # 计算因子值
        output.loc[:,new_industry_list] = self.gen_normal_factor(
            factor.loc[:,new_industry_list], factor_name,"rolling",window = 250*5)        
        
        for fn in ["银行","证券Ⅱ"]:
            
            temp = factor[fn]
            
            # 获取不包含nan的数据
            last_nan = temp[temp.isnull()].index[-1]
            app_data = temp.iloc[temp.index.get_loc(last_nan)+1:]
            
            # 计算分位数
            output.loc[app_data.index,fn] = self.gen_normal_factor(app_data,"temp","rolling",window = 250*5)
        
        # 调整方向（截面kurt和时序上不太一样）
        inv_list = ["down_pct","limitdown_pct",    # 股票下跌数目越少越拥挤
                    "comp_ret_vol","comp_ret_skewness", # 分化度越小越拥挤
                    "comp_turn_vol","comp_turn_skewness",]
        
        # inv_list = ["down_pct","limitdown_pct",    # 股票下跌数目越少越拥挤
        #             "comp_ret_vol","comp_ret_skewness","comp_ret_kurtosis",  # 分化度越小越拥挤
        #             "comp_turn_vol","comp_turn_skewness","comp_turn_kurtosis"]
        
        if factor_name in inv_list:
            output = 1 - output
            
        output = updown_restriction(updown = 0,factors = output)
        
        return output

    
    def updown_restriction(self, factors, updown = 1):
        factor = factors.copy()
    
        MA_60 = self.index_daily_ret.rolling(window = 60,min_periods = 10).mean()
        MA_5 = self.index_daily_ret.rolling(window = 5, min_periods = 2).mean()
        MA_20 = self.index_daily_ret.rolling(window = 20,min_periods = 5).mean()
        
        if updown == 1:
            a = self.index_daily_ret > 0
            b = self.index_daily_ret < MA_5
            c = self.index_daily_ret > MA_60  
        elif updown == -1:
            a = self.index_daily_ret < 0
            b = self.index_daily_ret > MA_5
            c = self.index_daily_ret < MA_60
            
        else:
            return factor
        
        factor[a & b & c] == np.nan
        return factor


if __name__ == "__main__":
    
    model = crowd_factors()

    path = "./factor"
    
    # 动量
    name_all = ['normal_momentum','sharpe_momentum','distance_momentum'
                'information_momentum','tail_momentum']
    
    for factor_name in name_all:
        for window in [5,10,20,40,60]:
            print(factor_name,window)
            factor = model.gen_indus_momentum(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))

    # 流动性
    name_all = ["turn",'volume','amount']
    
    for factor_name in name_all:
        for window in [5,10,20,40,60]:
            print(factor_name,window)
            factor = model.gen_indus_flow_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))
            
    # 乖离率
    name_all = ["close_bias","turn_bias",'volume_bias']
    
    for factor_name in name_all:
        for window in [20,40,60,120,250]:
            print(factor_name,window)
            factor = model.gen_indus_bias_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))

    # 滚动相关系数
    name_all = ["corr_volume_close","corr_amount_close","corr_turn_close"]

    for factor_name in name_all:
        for window in [20,40,60]:
            print(factor_name,window)
            factor = model.gen_roll_corr_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))


    # 波动率
    name_all = ["vol","downvol"]
    
    for factor_name in name_all:
        for window in [5,10,20,40,60]:
            print(factor_name,window)
            factor = model.gen_indus_vol_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))

            
    # 分布特征
    name_all = ['kurtosis','skewness','coskewness']
    
    for factor_name in name_all:
        for window in [5,10,20,40,60]:
            print(factor_name,window)        
            factor = model.gen_indus_distribution_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))
                            
    # 上涨比例、涨停比例
    name_all = ["up_pct",'down_pct',"limitup_pct",'limitdown_pct']
    
    for factor_name in name_all:
        for window in [1,5,10,20]:
            print(factor_name,window)
            factor = model.gen_updown_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))
                

    # # 成分股涨跌/换手率分布特征
    name_all = ["comp_ret_vol",'comp_ret_skewness',"comp_ret_kurtosis",
                "comp_turn_vol",'comp_turn_skewness','comp_turn_kurtosis']
    
    for factor_name in name_all:
        for window in [1,5,10,20]:
            print(factor_name,window)
            factor = model.gen_comp_distribution_factor(factor_name,window)
            factor.to_csv('{}/{}_{}.csv'.format(path,factor_name,window))
            