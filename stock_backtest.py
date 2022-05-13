import pandas as pd
import numpy as np

from utils import BacktestUtils
from utils import PerfUtils

class IndustryEvent:

    def __init__(self):
    
        # 读取交易日数据
        self.daily_dates = pd.read_pickle("./data/basic/daily_dates")
        
        # 读取股票收盘价数据
        self.stock_close = pd.read_pickle("./data/daily/stock/S_DQ_ADJCLOSE")
        
        # 读取股票换手率
        self.turn = pd.read_pickle("./data/daily/stock/S_DQ_TURN")
                
        # 读取基准指数收盘价数据
        self.index_close = pd.read_pickle("./data/daily/index/index_close")
        
        # 读取股票预处理相关数据
        self.updown = pd.read_pickle("./data/daily/stock/UP_DOWN_LIMIT_STATUS")
        self.trade_st = pd.read_pickle("./data/daily/stock/S_DQ_TRADESTATUS")
        self.st_mark = pd.read_pickle("./data/daily/stock/ST_mark")
        self.listed_days  = pd.read_pickle("./data/daily/stock/listed_days")
        
        # 过滤器：滤除涨跌停、换手率为0、非交易状态、ST状态、上市天数小于180的股票
        # 股票持有时间比较长，不能剔除涨跌停
        self.stock_filter = (self.updown.abs() != 1) & (self.turn > 1e-8) & \
                            (self.trade_st == "交易") & (self.st_mark != 1) & \
                            (self.listed_days >= pd.Timedelta("180 days"))
                            
                            
    # =========================================================================
    # 生成调仓信号           
    # =========================================================================
    def portfolio_gen(self,df_name):
                        
        # 调研信号
        survey_sum = pd.read_pickle('./results/'+df_name)
        
        # 填充空值
        survey_sum[survey_sum.isnull()] = 0
                
        # 将数据向后平移，即认为机构调研数据有一定的滞后性
        #survey_sum_rolling = survey_sum.rolling(holding_days).max()
                     

        # 当行业指标给定的最后日期大于最后一个交易日时, 踢除无法调仓的最后交易信号（月末容易出现）
        if survey_sum.index[-1] >= self.daily_dates[-1]:
            survey_sum = survey_sum.drop(index=survey_sum.index[-1])
            
        # 按照之前的交易日序列计算汇总信号数据时对应的最后一个交易日
        survey_sum.index = \
        [self.daily_dates[self.daily_dates > i].index[0] for i in survey_sum.index] 
        
        # 生成具体持仓信息
        signal_adj = survey_sum.replace(0, np.nan)
            
        # 信号过滤，剔除异常股票
        # 如果上一个持仓日持有该股票，不需要卖出，只对买入股票进行判断
        signal_adj[(~self.stock_filter.loc[signal_adj.index, signal_adj.columns]) &
                   (signal_adj.shift(1) == False)] = False
        
        
        signal_adj = (signal_adj.rank(axis = 1).T - signal_adj.rank(axis = 1).median(axis = 1)).T
        
        # 填充空值
        signal_adj[signal_adj.isnull()] = 0
        
        # 日期调整
        self.signal_adj_f = signal_adj
        return signal_adj
    
        
    # =============================================================================
    #  回测主程序
    # =============================================================================
    def backtest(self, df_name,arg):
        
        # 生成调仓信号
        df = self.portfolio_gen(df_name)
        indus_num = arg['pre_num']
        # 按日期截断
        df = df.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 提取沪深300指数作为基准
        base_nav = self.index_close.loc[arg['回测开始时间']:, '000985.CSI']
        stock_close = self.stock_close.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 初始化返回值
        merge_ls_factor = df.copy() * 0
        merge_factor_rank = df.rank(method='average', ascending=False, axis=1)       
        # 多头持仓行业
        long_judge = (merge_factor_rank <= indus_num)
        long_judge[long_judge.sum(axis=1) > indus_num] = \
            (merge_factor_rank <= indus_num - 1)[long_judge.sum(axis=1) > indus_num] 
        merge_ls_factor[long_judge] = 1

        # 加权后的景气度指标进行排序 - 由小到大
        merge_factor_rank = df.rank(method='average', ascending=True, axis=1)

        # 空头持仓行业
        short_judge = (merge_factor_rank <= indus_num)
        short_judge[short_judge.sum(axis=1) > indus_num] = \
            (merge_factor_rank <= indus_num - 1)[short_judge.sum(axis=1) > indus_num] 
        merge_ls_factor[short_judge] = - 1
        
        merge_ls_factor[merge_ls_factor.isnull()] = 0        
        
        # 根据输入的行业指标，计算多头和空头持仓
        df_factor = merge_ls_factor.copy()
        long_portion = df_factor.copy() * 0
        short_portion = df_factor.copy() * 0
        
        # 遍历所有日期
        for date in df_factor.index:    
            
            # 一般情况，标为1的行业为多头持仓，这里简化为指标值大于零的行业为多头持仓
            if sum(df_factor.loc[date,:] > 0) != 0:
                long_portion.loc[date,df_factor.loc[date,:]>0] = 1/sum(df_factor.loc[date,:] > 0)
                
            # 一般情况，标为-1的行业为空头持仓，这里简化为指标值小于零的行业为空头持仓
            if sum(df_factor.loc[date,:] < 0) != 0:
                short_portion.loc[date,df_factor.loc[date,:]<0] = 1/sum(df_factor.loc[date,:] < 0)

        # 计算绝对净值        
        nav = pd.DataFrame(columns=['多头策略','空头策略','基准'])
        
        # 回测,计算策略净值
        nav['多头策略'], df_indus_return = BacktestUtils.cal_nav(long_portion, \
           stock_close, base_nav, arg['手续费'])
            
        nav['空头策略'], df_indus_return_short = BacktestUtils.cal_nav(short_portion, \
           stock_close, base_nav, arg['手续费'])
        
        # 基准净值归一化
        nav['基准'] = base_nav / base_nav.values[0]
          
        # 计算相对净值
        nav_relative = pd.DataFrame(columns=['多头/基准','空头/基准'])
        nav_relative['多头/基准'] = nav['多头策略'] / nav['基准'] 
        nav_relative['空头/基准'] = nav['空头策略'] / nav['基准']

                    
        # 计算月度调仓信号，用于计算胜率
        nav_resample = nav.resample('M').last()
        monthly_dates = self.daily_dates.resample('M').last()
        panel_dates = pd.to_datetime(monthly_dates[nav_resample.index].values)
        
        # 计算策略表现
        perf_long = PerfUtils.excess_statis(nav['多头策略'], nav['基准'], panel_dates)
        perf_short = PerfUtils.excess_statis(nav['空头策略'], nav['基准'], panel_dates)           
        # 给出序号的时候进行存储操作


        return nav, nav_relative, long_portion, short_portion, df_indus_return

    # =============================================================================
    #  分层回测主程序
    # =============================================================================        
        
    def clsfy_backtest(self, df_name,arg):
        
        
        df = portfolio_gen(df_name).replace(0,np.nan)

        # 按日期截断
        df = df.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 提取沪深300指数作为基准
        base_nav = self.index_close.loc[arg['回测开始时间']:, '000985.CSI']
        stock_close = self.stock_close.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 初始化返回值
        merge_factor = df.copy() * 0
        merge_factor_rank = df.rank(method='average', ascending=False, axis=1,numeric_only = True)       


        # 分层数
        indus_number = 5
        layer_ind_num = np.int(np.floor(merge_factor_rank.shape[1] / indus_number))
        inter_part = merge_factor_rank.shape[1] % indus_number / indus_number
        thres = layer_ind_num + inter_part
        
        # 初始化返回值
        factor_1 = merge_factor.copy() * 0
        factor_2 = merge_factor.copy() * 0
        factor_3 = merge_factor.copy() * 0
        factor_4 = merge_factor.copy() * 0
        factor_5 = merge_factor.copy() * 0
               
        # 多头持仓行业
        factor_1[merge_factor_rank <= thres] = 1
        factor_2[(merge_factor_rank > thres) & (merge_factor_rank <= 2 * thres)] = 1
        factor_3[(merge_factor_rank > 2 * thres) & (merge_factor_rank <= 3 * thres)] = 1
        factor_4[(merge_factor_rank > 3 * thres) & (merge_factor_rank <= 4 * thres)] = 1
        factor_5[merge_factor_rank > 4 * thres] = 1
        
        if np.ceil(thres) != np.floor(thres):
            factor_1[merge_factor_rank == np.ceil(thres)] = thres - np.floor(thres)
            factor_2[merge_factor_rank == np.ceil(thres)] = np.ceil(thres) - thres

        if np.ceil(2 * thres) != np.floor(2 * thres):
            factor_2[merge_factor_rank == np.ceil(2 *thres)] = 2 * thres - np.floor(2 * thres)
            factor_3[merge_factor_rank == np.ceil(2 *thres)] = np.ceil(2 * thres) - 2 * thres
            
        if np.ceil(3 * thres) != np.floor(3 * thres):
            factor_3[merge_factor_rank == np.ceil(3 *thres)] = 3 * thres - np.floor(3 * thres)
            factor_4[merge_factor_rank == np.ceil(3 *thres)] = np.ceil(3 * thres) - 3 * thres
            
        if np.ceil(4 * thres) != np.floor(4 * thres):
            factor_4[merge_factor_rank == np.ceil(4 *thres)] = 4 * thres - np.floor(4 * thres)
            factor_5[merge_factor_rank == np.ceil(4 *thres)] = np.ceil(4 * thres) - 4 * thres
            
        # 计算多头持仓
        factor_1 = (factor_1.T / factor_1.sum(axis = 1)).T
        factor_2 = (factor_2.T / factor_2.sum(axis = 1)).T
        factor_3 = (factor_3.T / factor_3.sum(axis = 1)).T
        factor_4 = (factor_4.T / factor_4.sum(axis = 1)).T
        factor_5 = (factor_5.T / factor_5.sum(axis = 1)).T
        
                        
        # 计算绝对净值        
        nav = pd.DataFrame(columns=['分层1','分层2','分层3','分层4','分层5','基准'])
        
        # 回测,计算策略净值
        nav['分层1'], _ = BacktestUtils.cal_nav(factor_1, stock_close, base_nav, fee=arg['手续费'])
        nav['分层2'], _ = BacktestUtils.cal_nav(factor_2, stock_close, base_nav, fee=arg['手续费'])
        nav['分层3'], _ = BacktestUtils.cal_nav(factor_3, stock_close, base_nav, fee=arg['手续费'])
        nav['分层4'], _ = BacktestUtils.cal_nav(factor_4, stock_close, base_nav, fee=arg['手续费'])
        nav['分层5'], _ = BacktestUtils.cal_nav(factor_5, stock_close, base_nav, fee=arg['手续费'])
        
        # 基准净值归一化
        nav['基准'] = base_nav / base_nav.values[0]
          
        # 计算相对净值
        nav_relative = pd.DataFrame(columns=['分层1','分层2','分层3','分层4','分层5'])
        nav_relative['分层1'] = nav['分层1'] / nav['基准'] 
        nav_relative['分层2'] = nav['分层2'] / nav['基准'] 
        nav_relative['分层3'] = nav['分层3'] / nav['基准'] 
        nav_relative['分层4'] = nav['分层4'] / nav['基准'] 
        nav_relative['分层5'] = nav['分层5'] / nav['基准'] 
        
        # 返回绝对净值曲线，相对净值曲线，多头持仓
        return nav, nav_relative, factor_1        

    
    def clsfy_backtest2(self, df_name,arg):
        
        
        df = self.portfolio_gen(df_name)
        df = df.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 提取沪深300指数作为基准
        base_nav = self.index_close.loc[arg['回测开始时间']:, '000985.CSI']
        stock_close = self.stock_close.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        # 初始化返回值
        merge_factor = df.replace(0,np.nan).copy()

        # 分层数
        indus_number = 5
        thres1 = merge_factor.quantile(0.2,axis = 1,numeric_only=True)
        thres2 = merge_factor.quantile(0.4,axis = 1,numeric_only=True)
        thres3 = merge_factor.quantile(0.6,axis = 1,numeric_only=True)
        thres4 = merge_factor.quantile(0.8,axis = 1,numeric_only=True)
        
        # 初始化返回值
        factor_1 = df.copy() * 0
        factor_2 = df.copy() * 0
        factor_3 = df.copy() * 0
        factor_4 = df.copy() * 0
        factor_5 = df.copy() * 0
               
        # 多头持仓行业

        factor_1[(merge_factor.T <= thres1).T] = 1
        factor_2[((merge_factor.T > thres1) & (merge_factor.T <= thres2)).T] = 1
        factor_3[((merge_factor.T > thres2) & (merge_factor.T <= thres3)).T] = 1
        factor_4[((merge_factor.T > thres3) & (merge_factor.T <= thres4)).T] = 1
        factor_5[(merge_factor.T > thres4).T] = 1
                    
        # 计算多头持仓
        factor_1 = (factor_1.T/factor_1.sum(axis = 1)).T
        factor_2 = (factor_2.T/factor_2.sum(axis = 1)).T
        factor_3 = (factor_3.T/factor_3.sum(axis = 1)).T
        factor_4 = (factor_4.T/factor_4.sum(axis = 1)).T
        factor_5 = (factor_5.T/factor_5.sum(axis = 1)).T
        
                        
        # 计算绝对净值        
        nav = pd.DataFrame(columns=['分层1','分层2','分层3','分层4','分层5','基准'])
        
        # 回测,计算策略净值
        nav['分层1'], _ = BacktestUtils.cal_nav(factor_1, stock_close, base_nav, fee=arg['手续费'])
        nav['分层2'], _ = BacktestUtils.cal_nav(factor_2, stock_close, base_nav, fee=arg['手续费'])
        nav['分层3'], _ = BacktestUtils.cal_nav(factor_3, stock_close, base_nav, fee=arg['手续费'])
        nav['分层4'], _ = BacktestUtils.cal_nav(factor_4, stock_close, base_nav, fee=arg['手续费'])
        nav['分层5'], _ = BacktestUtils.cal_nav(factor_5, stock_close, base_nav, fee=arg['手续费'])
        
        # 基准净值归一化
        nav['基准'] = base_nav / base_nav.values[0]
          
        # 计算相对净值
        nav_relative = pd.DataFrame(columns=['分层1','分层2','分层3','分层4','分层5'])
        nav_relative['分层1'] = nav['分层1'] / nav['基准'] 
        nav_relative['分层2'] = nav['分层2'] / nav['基准'] 
        nav_relative['分层3'] = nav['分层3'] / nav['基准'] 
        nav_relative['分层4'] = nav['分层4'] / nav['基准'] 
        nav_relative['分层5'] = nav['分层5'] / nav['基准'] 
        
        # 返回绝对净值曲线，相对净值曲线，多头持仓
        return nav, nav_relative, factor_1 
    

    def calc_ic(self,df_name,arg):
        df_survey = pd.read_pickle('./results/'+df_name)
        data = df_survey.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        close = self.stock_close.loc[arg['回测开始时间']:arg['回测结束时间'],:]
        
        data = data[self.stock_filter].fillna(0)
        close_p = close.reindex(index = data.index,columns = data.columns)
        close_p = close_p[self.stock_filter]
        close_p = (close_p.shift(-1)- close_p)/close_p
        close_p = close_p.fillna(0)

        perff = pd.DataFrame(np.nan, index = close_p.index,columns = ['ic','ic_long','ic_short',\
                    'ic_1','ic_2','ic_3','ic_4','ic_5'])
        for j in close_p.index:
            data_slice = data.loc[j]
            close_slice = close_p.loc[j]
            ic = np.corrcoef(close_slice, data_slice)[0,1]
            perff.loc[j,'ic'] = ic

            data_up = data_slice[data_slice>0]
            close_up = close_slice[data_slice>0]
            data_down = data_slice[data_slice<0]
            close_down = close_slice[data_slice<0]

            ic_long = np.corrcoef(close_up, data_up)[0,1]
            ic_short = np.corrcoef(close_down, data_down)[0,1]
            perff.loc[j,'ic_long'] = ic_long
            perff.loc[j,'ic_short'] = ic_short

            th1 = data_slice.quantile(0.2)
            th2 = data_slice.quantile(0.4)
            th3 = data_slice.quantile(0.6)
            th4 = data_slice.quantile(0.8)
            data_1 = data_slice[data_slice<th1]
            close_1 = close_slice[data_slice<th1]
            perff.loc[j,'ic_1'] = np.corrcoef(close_1, data_1)[0,1]
            data_2 = data_slice[(th1<data_slice) & (data_slice<=th2)]
            close_2 = close_slice[(th1<data_slice) & (data_slice<=th2)] 
            perff.loc[j,'ic_2'] = np.corrcoef(close_2, data_2)[0,1]        
            data_3 = data_slice[(th2<data_slice) & (data_slice<=th3)]
            close_3 = close_slice[(th2<data_slice) & (data_slice<=th3)] 
            perff.loc[j,'ic_3'] = np.corrcoef(close_3, data_3)[0,1]   
            data_4 = data_slice[(th3<data_slice) & (data_slice<=th4)]
            close_4 = close_slice[(th3<data_slice) & (data_slice<=th4)] 
            perff.loc[j,'ic_4'] = np.corrcoef(close_4, data_4)[0,1]   
            data_5 = data_slice[th4<data_slice]
            close_5 = close_slice[th4<data_slice] 
            perff.loc[j,'ic_5'] = np.corrcoef(close_5, data_5)[0,1]   

        perff_cum = perff.cumsum()
        return perff_cum
    
    
if __name__ == "__main__":
            
    # 模型初始化
    
    model = IndustryEvent()
    arg =   {'pre_num':100,
            '回测开始时间': '2015-01-01',
            '回测结束时间': '2021-02-28',  '手续费': 0.000}          
    # 测试
    file_list = [i for i in os.listdir('./results/') if i[0] != '.']
    result_df = pd.DataFrame()
    for i in file_list:
        print(i)
        nav, nav_relative, _, _, _  = model.backtest(i,arg)
        result_df.loc[i,'多头策略'] = nav['多头策略'].iloc[-1]
        result_df.loc[i,'空头策略'] = nav['空头策略'].iloc[-1] 
        result_df.loc[i,'基准'] = nav['基准'].iloc[-1]    
        
        result_df.loc[i,'多头/基准'] = nav_relative['多头/基准'].iloc[-1]
        result_df.loc[i,'空头/基准'] = nav_relative['空头/基准'].iloc[-1] 
        
        
        nav_clsfy, nav_clsfy_relative, _  = model.clsfy_backtest2(i,arg)
        result_df.loc[i,'分层1'] = nav_clsfy['分层1'].iloc[-1]
        result_df.loc[i,'分层2'] = nav_clsfy['分层2'].iloc[-1] 
        result_df.loc[i,'分层3'] = nav_clsfy['分层3'].iloc[-1]
        result_df.loc[i,'分层4'] = nav_clsfy['分层4'].iloc[-1] 
        result_df.loc[i,'分层5'] = nav_clsfy['分层5'].iloc[-1]
        result_df.loc[i,'分层1/基准'] = nav_clsfy_relative['分层1'].iloc[-1]
        result_df.loc[i,'分层2/基准'] = nav_clsfy_relative['分层2'].iloc[-1] 
        result_df.loc[i,'分层3/基准'] = nav_clsfy_relative['分层3'].iloc[-1]
        result_df.loc[i,'分层4/基准'] = nav_clsfy_relative['分层4'].iloc[-1] 
        result_df.loc[i,'分层5/基准'] = nav_clsfy_relative['分层5'].iloc[-1]
        
        
        perff_cum = model.calc_ic(i,arg)
        result_df.loc[i,'ic_long'] = perff_cum['ic_long'].iloc[-2]
        result_df.loc[i,'ic_short'] = perff_cum['ic_short'].iloc[-2] 
        result_df.loc[i,'ic'] = perff_cum['ic'].iloc[-2]     

    

