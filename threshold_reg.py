# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import scipy.stats
import empyrical

plt.style.use("seaborn-white")
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class model_regress():
    
    # -------------------------------------------------------------------------
    # 实例化，加载基本信息
    # -------------------------------------------------------------------------
    def __init__(self):
                            
        # 读取行业指数列表
        industry_list = pd.read_pickle('./data/indus_info')['行业名称'].tolist()
        self.industry_list = [i for i in industry_list if i not in ["综合金融","多元金融","保险Ⅱ"]]
        
        # 获取一级行业收盘价
        self.index_close = pd.read_pickle("./data/Wind_indus_close")
        self.index_close = self.index_close.reindex(columns=self.industry_list)
                                                            
        # 生成日频交易序列
        self.daily_dates = pd.Series(self.index_close.index, index=self.index_close.index)
       
        # 绘图颜色设置
        self.color = np.array([[0.75294118, 0.        , 0.        ],
                               [0.01176471, 0.01176471, 0.01176471],
                               [0.89411765, 0.42745098, 0.04313725],
                               [0.21568627, 0.37647059, 0.57254902],
                               [0.49803922, 0.49803922, 0.49803922]])
                    
    # -------------------------------------------------------------------------
    # 门限回归计算
    # [输入]
    # thres_delta     间隔阈值设定
    # start_time      计算起始时间
    # end_time        计算终止时间
    # factor          拥挤度指标
    # -------------------------------------------------------------------------
    def reg_results(self, thres_delta, start_time, end_time, factor):
        
        threshold = np.arange(0.5, 0.95 + thres_delta, thres_delta)
        output = pd.DataFrame(index=threshold,columns=["k","p","r2","neg"])
        
        # 计算收益率
        indus_return = self.index_close.shift(-21) / self.index_close.shift(-1) - 1
        result = self.index_close / self.index_close.shift(1) - 1
        indus_max_drawdown = result.shift(-20).rolling(window = 20).apply(lambda x: empyrical.max_drawdown(x), raw= True)
        y_temp = indus_max_drawdown.loc[start_time:end_time,:]
        x_temp = factor.loc[y_temp.index]
        
        for th in threshold:
            
            # 数据合并为一列
            x_s = x_temp.values[~np.isnan(y_temp)].reshape(-1,1)
            y_s = y_temp.values[~np.isnan(y_temp)].reshape(-1,1)
                        
            # 设定门限值
            y_s = y_s[x_s>=th]
            x_s = x_s[x_s>=th]
            
            if len(x_s) == 0:
                continue 
            
            # 去掉离群值
            ub = y_s.mean()+3*y_s.std()
            lb = y_s.mean()-3*y_s.std()
            x_s = x_s[(y_s>=lb) & (y_s<=ub)]
            y_s = y_s[(y_s>=lb) & (y_s<=ub)]
                        
            # 线性回归
            results = sm.OLS(y_s,sm.add_constant(x_s)).fit()
            
            # Newey-West调整
            nw_results = results.get_robustcov_results(cov_type='HAC',maxlags=1)
            try:
                output.loc[th,"k"] = nw_results.params[1]
                output.loc[th,"p"] = nw_results.pvalues[1]
                output.loc[th,"r2"] = nw_results.rsquared
                output.loc[th,"neg"] = (y_s > 0).sum() / len(y_s)
                output.loc[th,"return"] = np.median(y_s)
            except:
                output.loc[th,"k"] = np.nan
                output.loc[th,"p"] = np.nan
                output.loc[th,"r2"] = np.nan
                output.loc[th,"neg"] = (y_s > 0).sum() / len(y_s)
                output.loc[th,"return"]= np.median(y_s)
        return output
    
    # -------------------------------------------------------------------------
    # 散点图绘制
    # [输入]
    # nparts          分块数
    # start_time      计算起始时间
    # end_time        计算终止时间
    # factor          拥挤度指标
    # -------------------------------------------------------------------------
    def reg_scatter(self, nparts, start_time, end_time, factor):

        # 计算收益率
        indus_return = self.index_close.shift(-21) / self.index_close.shift(-1) - 1
        result = self.index_close / self.index_close.shift(1) - 1
        indus_max_drawdown = result.shift(-20).rolling(window = 20).apply(lambda x: empyrical.max_drawdown(x), raw= True)
        y_temp = indus_max_drawdown.loc[start_time:end_time,:][factor>=0]
        x_temp = factor.loc[y_temp.index][factor>=0]
        
        # 展开
        x = x_temp.values[~np.isnan(x_temp)].reshape(-1,1)
        y = y_temp.values[~np.isnan(x_temp)].reshape(-1,1)

        # scatter
        fig,ax = plt.subplots(figsize=(15,6))
        for i,th in enumerate(np.arange(0,1,1/nparts)):
            cond1 = (x>=th)&(x<th+1/nparts)
            cond2 = ((y[cond1]-y[cond1].mean())<=3*y[cond1].std()) & ((y[cond1]-y[cond1].mean())>=-3*y[cond1].std())
            ax.scatter(x[cond1][cond2], y[cond1][cond2], alpha=0.5, color=self.color[0]+[0,0.1,0.1])
            ax.set_title("N = "+str(nparts),fontsize=20)
    
    
    def reg_bar(self, output):
        
        # bar
        # fig = plt.figure(figsize=(15,4))
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(2,2,1)
        ax.bar(output.index,output["k"], color=self.color[0], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("k",fontsize=20)

        ax = fig.add_subplot(2,2,2)
        ax.bar(output.index,output["p"], color=self.color[3], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("p",fontsize=20)
        
        ax = fig.add_subplot(2,2,3)
        ax.bar(output.index,output["neg"], color=self.color[3], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("neg",fontsize=20)
        
        ax = fig.add_subplot(2,2,4)
        ax.bar(output.index,output["return"], color=self.color[3], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("return",fontsize=20)
        
        # ax = fig.add_subplot(1,2,2)
        # ax.bar(output.index,output["p"], color=self.color[3], width=0.3/output.shape[0])
        # ax.tick_params(labelsize=20)
        # ax.set_title("p",fontsize=20)
    
    # -------------------------------------------------------------------------
    # 显示进度条
    # [输入]
    # length   总长度
    # step     当前步数
    # -------------------------------------------------------------------------    
    def process(self, length, step):

        pct = 100*(step+1)/length
        p1 = "#"*int(pct/2)
        p2 = "-"*(50-int(pct/2))
        print("\r %6.2f%% | [%s%s] | %d/%d" % (pct,p1,p2,step+1,length),end="")
                
    # -------------------------------------------------------------------------
    # 将给定路径下的指标进行叠加
    # [输入]
    # path        需要进行复合的单项行业景气度指标存储路径
    # factors     进行复合的行业指标名称
    # cut_index   输入对于指标进行截断的时间点
    # -------------------------------------------------------------------------
    def cal_merge_factor(self, path, factors, cut_index):
        
        # 复合指标记录变量初始化
        m_factor = None

        # 遍历给定的单项行业景气度指标   
        for factor_name in factors:
            
            # 读取单项行业景气度指标
            df = pd.read_pickle(path + factor_name)
            df[df.isnull()] = 0
            df_abs = pd.DataFrame(np.ones_like(df.values),index=df.index,columns=df.columns)
            
            # 将不同单项景气度指标进行叠加
            if m_factor is None:
                m_factor = df
                m_factor_abs = df_abs
                
            else:
                # 不同单项景气度指标时间起点有区别
                m_index = m_factor.index if len(df.index) > len(m_factor.index) else df.index
                m_factor.loc[m_index,:] = m_factor.loc[m_index,:] + df.loc[m_index,:]    
                m_factor_abs.loc[m_index,:] = m_factor_abs.loc[m_index,:] + df_abs.loc[m_index,:]    
                
        return m_factor.loc[cut_index:,:] / m_factor_abs.loc[cut_index:,:] 
      
    # -------------------------------------------------------------------------
    # 将给定路径下的指标进行叠加
    # [输入]
    # path        需要进行复合的单项行业景气度指标存储路径
    # factors     进行复合的行业指标名称
    # cut_index   输入对于指标进行截断的时间点
    # -------------------------------------------------------------------------
    def cal_add_factor(self, path, factors, cut_index):
        
        # 复合指标记录变量初始化
        m_factor = []
        
        # 遍历给定的单项行业景气度指标   
        for factor_name in factors:
            
            # 读取单项行业景气度指标
            crowd_factor_origin = pd.read_pickle(path + factor_name)
            
            crowd_factor_origin = crowd_factor_origin.reindex(columns=self.industry_list)
            crowd_factor_origin = crowd_factor_origin.reindex(index=self.daily_dates)
                        
            m_factor.append(crowd_factor_origin)     
            
        # 指标合成
        merge_factor = pd.DataFrame(index=m_factor[0].index, columns=m_factor[0].columns)
        
        for column in m_factor[0].columns:
            
            factor = []
            
            for index in range(0,len(factors)):
                
                factor.append(m_factor[index].loc[:,column])
            
            # merge_factor.loc[:,column] = pd.concat(factor, axis=1).median(axis=1)            
            merge_factor.loc[:,column] = pd.concat(factor, axis=1).max(axis=1)
            
        return merge_factor
                


if __name__ == '__main__':               
# =============================================================================
#   门限回归 - 单个指标门限回归分析
# =============================================================================

    # 模型初始化
    model = model_regress()

    start_time = '2010-01-01'
    # start_time = '2017-01-01'
    end_time = '2020-09-23'
    
    # 读取指标数据
    # crowd_factor = pd.read_pickle('./factor/'+'comp_turn_kurtosis_10')
    # crowd_factor = pd.read_pickle('./factor/'+'limitup_pct_20')
    crowd_factor = pd.read_pickle('./factor/'+'turn_40')
    # crowd_factor = pd.read_pickle('./factor/'+'corr_amount_close_40')
    # crowd_factor = pd.read_pickle('./factor/'+'kurtosis_40')
    
    # # 设定负数指标
    # crowd_factor[model.index_close.pct_change(20)<0] *= -1
    
    
    # 回测结果
    results = model.reg_results(0.01, start_time, end_time, crowd_factor)
        
    # 散点图绘制
    model.reg_scatter(10, start_time, end_time, crowd_factor)
    
    # 回归柱状图分析
    model.reg_bar(results)
    plt.show()
    
# #     # # 计算收益率
# #     # factor = crowd_factor
# #     # indus_return = model.index_close.shift(-21) / model.index_close.shift(-1) - 1
# #     # y_temp = indus_return.loc[start_time:end_time,:][factor>=0]
# #     # x_temp = factor.loc[y_temp.index][factor>=0]
    
# #     # # 展开
# #     # x = x_temp.values[~np.isnan(x_temp)].reshape(-1,1)
# #     # y = y_temp.values[~np.isnan(x_temp)].reshape(-1,1)

# #     # y = y[x>=0.5]
# #     # x = x[x>=0.5]

# #     # # 去掉离群值
# #     # ub = y.mean()+3*y.std()
# #     # lb = y.mean()-3*y.std()
# #     # x = x[(y>=lb) & (y<=ub)]
# #     # y = y[(y>=lb) & (y<=ub)]    
        
# #     # data = []
    
# #     # for th in [0.7, 0.8, 0.9, 0.95]:
        
# #     #     # 设定门限值
# #     #     y = y[x>=th]
# #     #     x = x[x>=th]
                    
# #     #     # Decide the XY position, and calculate the Gaussian distribution
# #     #     # positions = np.vstack([X.ravel(), Y.ravel()])
# #     #     kernel = scipy.stats.gaussian_kde(y)
# #     #     a = kernel(np.arange(-0.6,0.6,0.01).T)

# #     #     data.append(a)
# #     # b = np.vstack(data).T
    
# # =============================================================================
# #   门限回归 - 遍历测试
# # =============================================================================

#     # 模型初始化
#     model = model_regress()

#     # start_time = '2010-01-01'
#     # end_time = '2020-9-23'

#     start_time = '2016-01-01'
#     end_time = '2020-9-23'
    
#     # 读取指标列表
#     factor_name = os.listdir(os.path.join(os.getcwd(),"factor"))

#     # 门限回归评价指标计算
#     perf = pd.DataFrame(index=["pct_k<0","corr1","corr2","corr3","pct_p<0.01", 'K', 'neg'])

#     for index in range(0, len(factor_name)):
        
#         # 进度条
#         model.process(len(factor_name), index)
        
#         # 读取指标数据
#         crowd_factor = pd.read_pickle('factor/' + factor_name[index])
        
#         # 设定负数指标
#         crowd_factor[model.index_close.pct_change(20)<0] *= -1
        
#         # 0.025用来调整门限值的数量
#         results = model.reg_results(0.01, start_time, end_time, crowd_factor)
#         results["nseq"] = results.index
#         results = results.astype(float)
        
#         # 计算3个评价指标
#         perf[factor_name[index]] = [(results.loc[0.8:]["k"]<0).mean(),
#                                     results["k"].corr(results["nseq"], method="pearson"),
#                                     results["neg"].corr(results["nseq"], method="pearson"),
#                                     results["return"].corr(results["nseq"], method="pearson"),                                    
#                                     ((results["p"]<0.1) & (results["k"]<0)).mean(),                                    
#                                     results.iloc[-1,0], results.iloc[-1,3]]
            
#     # 导出数据
#     perf.T.to_excel("回测统计/门限回归测试结果-2016-2020.xlsx")
#     # perf.T.to_excel("回测统计/门限回归测试结果-2010-2020.xlsx")

#     # 读取门限回归测试结果
#     # ret = pd.read_excel("回测统计/门限回归测试结果-2010-2020.xlsx", index_col=0)
#     ret = pd.read_excel("回测统计/门限回归测试结果-2016-2020.xlsx", index_col=0)
#     # ret.columns = ['k>0.7时小于0的比例', 'K值相关系数', 'P值小于0.01的比例','最大位置K','负收益占比']
#     ret.columns = ['负K值比例', 'K值相关系数', '胜率相关系数', '收益率相关系数','显著性','最大位置K','负收益占比']
    
#     # judge_1 = ret['负K值比例'] >= 0.8
#     judge_2 = ret['K值相关系数'] < -0.5
#     judge_3 = ret['胜率相关系数'] < -0.5
#     judge_4 = ret['收益率相关系数'] < -0.5     
#     judge_5 = ret['显著性'] >= 0.7
    
#     judge_6 = pd.concat([judge_2, judge_3, judge_4, judge_5],axis=1).sum(axis=1)
    # judge_6 = judge_1 & judge_2 & judge_3
    
    
    
# # =============================================================================
# #   门限回归 - 单个指标门限回归分析
# # =============================================================================

#     # 模型初始化
#     model = model_regress()

#     start_time = '2010-01-01'
#     end_time = '2020-9-23'
    
#     # 读取指标数据
#     # factors = ['comp_turn_skewness_1','corr_turn_close_40', 'downvol_20','turn_40']
#     # factors = ['corr_turn_close_40','turn_40', 'downvol_20']
#     # factors = ['comp_turn_skewness_1','downvol_20','turn_40','vol_20']
#     factors = ['comp_turn_skewness_1','downvol_20','turn_40']
    
#     # factors = ['close_bias_250','comp_turn_kurtosis_1','comp_turn_skewness_1',
#     #            'corr_amount_close_40','downvol_20','limitup_pct_10','turn_60']
    
#     crowd_factor = model.cal_merge_factor('factor/', factors, cut_index='2006-01-01')
        
#     # 设定负数指标
#     crowd_factor[model.index_close.pct_change(20)<0] *= -1
    
#     # 回测结果
#     results = model.reg_results(0.025, start_time, end_time, crowd_factor)
        
#     # 散点图绘制
#     model.reg_scatter(50, start_time, end_time, crowd_factor)
    
#     # 回归柱状图分析
#     model.reg_bar(results)
#     plt.show()
        
        