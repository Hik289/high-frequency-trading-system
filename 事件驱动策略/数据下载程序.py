# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:46:35 2020


"""

import pandas as pd
import numpy as np
import pymongo


class data_model():
    
    # -------------------------------------------------------------------------
    # 加载数据库信息
    # -------------------------------------------------------------------------
    def __init__(self):
                
        # 获取MondoDB数据库链接
        self.client = pymongo.MongoClient("localhost", 27017)
        
        # 获取股票数据库对象
        self.stock_database = self.client["xquant_stock"]

        # 获取股票财报数据库对象
        self.stock_financial_database = self.client["xquant_stock_financial"]
        
        # 获取行业数据库对象
        self.indus_database = self.client["xquant_indus"]
        
        # 获取指数数据库对象
        self.index_database = self.client["xquant_index"]
        
        # 获取一致预取数据库对象
        self.est_database = self.client["xquant_est"]

        # 获取其他数据库对象
        self.other_database = self.client["xquant_other"]
        
        
    # -------------------------------------------------------------------------
    # 加载数据库信息
    # cursor      mongodb数据标签cursor
    # chunk_size  划分片数
    # -------------------------------------------------------------------------
    def cursor2dataframe(self, cursor, chunk_size: int):
        
        records = []  # 记录单片数据，写入dataframe
        frames = []   # 记录不同dataframe，拼接起来
        
        # 记录数据
        for i, record in enumerate(cursor):
            records.append(record)
            if i % chunk_size == chunk_size - 1:
                frames.append(pd.DataFrame(records))
                records = []
                
        # dataframe合并  
        if records:
                frames.append(pd.DataFrame(records))
                
        return pd.concat(frames)


    # -------------------------------------------------------------------------
    # 获取特定时间范围内的某个特定数据
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # date_name   日期类型
    # stock_name  股票类型
    # target      调取数据目标
    # -------------------------------------------------------------------------
    def get_specific_data(self, database, collection, start_date, end_date,
                             date_name, stock_name, target, code=None):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        if collection in ['AShareBalanceSheet','AShareCashFlow','AShareIncome']:
            
            # 读取408001000合并报表数据，部分被处罚公司此部分数据会被后来的公告修正
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}, 'STATEMENT_TYPE':408001000},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            
            # 读取数据, 存成DataFrame格式
            data = self.cursor2dataframe(cursor, 100000)
            data = data.pivot(index=date_name, columns=stock_name)[target]
        
            # 读取408005000合并报表(更正前)数据，此部分数据是公司最原始的数据
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}, 'STATEMENT_TYPE':408005000},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            
            # 读取数据, 存成DataFrame格式
            data_origin = self.cursor2dataframe(cursor, 100000)
            data_origin = data_origin.pivot(index=date_name, columns=stock_name)[target]
            
            # 数据替换，有旧数据的优先用旧数据
            data[~data_origin.isnull()] = data_origin
            
            # index重新改写
            data.index = pd.to_datetime(data.index)
            
            return data
        
        else:
            if code == None:
                cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            else:
                cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")},
                                stock_name:{"$in":code}},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            
            # 读取数据, 存成DataFrame格式
            data = self.cursor2dataframe(cursor, 100000)
        
            # 业绩预告数据容易出现重复
            if collection in ['AShareProfitNotice', 'HKIndexEODPrices', 'HKStockHSIndustriesMembers',
                              "AShareMoneyFlow","AShareEODDerivativeIndicator"]:
                
                # 相同股票数据去重
                data = data.sort_values(date_name)
                data.drop_duplicates(subset=[stock_name, date_name], keep='last', inplace=True) 
                
                
            # 重新整理数据index和columns
            data = data.pivot(index=date_name, columns=stock_name)[target]
                   
            # index重新改写
            data.index = pd.to_datetime(data.index)
            
            return data
        
    
    # -------------------------------------------------------------------------
    # 获取特定时间范围内的所有数据：
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # -------------------------------------------------------------------------
    def get_all_data(self, database, collection, start_date, end_date, date_name):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        cursor = db_collection.find({date_name:{
                            "$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")}},
                            {'_id':0}).sort('$natural',1)

        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 10000)
                
        return data

    
if __name__ == '__main__':
        
    model = data_model()
    
    start_date = '1988-01-01'
    end_date = '2021-12-31'

# =============================================================================
#  读取股票列表
# =============================================================================

    print("股票列表")
    
    # 最新A股信息
    stock_info = pd.DataFrame(list(model.stock_database["AShareDescription"].find({}, {'_id':0})))
    stock_info = stock_info.set_index('S_INFO_WINDCODE')
    
    # 剔除A和T开头的股票代码
    stock_info = stock_info.loc[[not i.startswith('A') for i in stock_info.index], :]
    stock_info = stock_info.loc[[not i.startswith('T') for i in stock_info.index], :]
    
    # 上市日期替换
    stock_info.loc[:,"S_INFO_LISTDATE"] = pd.to_datetime(stock_info.loc[:,"S_INFO_LISTDATE"])
        
    # 股票代码顺序重置
    stock_info.sort_index(inplace=True)

    

# =============================================================================
# index收盘价数据 - 用于策略回测
# =============================================================================

    print("指数收盘价")

    # 按照报告期读取数据
    codes = ['000300.SH', '000905.SH', '000008.SH', '000002.SH', '000985.CSI']
    index_close = model.get_specific_data(database = model.index_database,
                                        collection = 'AIndexEODPrices',
                                        start_date = start_date,
                                        end_date = end_date,
                                        date_name='TRADE_DT',           # 按照交易日（TRADE_DT）读取数据
                                        stock_name='S_INFO_WINDCODE',   # 股票代码
                                        target='S_DQ_CLOSE',
                                        code = codes)         # 读取收盘价数据
    
    index_close = index_close.reindex(columns=codes)
    index_close.to_pickle('data/daily/index/index_close')




# =============================================================================
# 下载股票数据
# =============================================================================
        
    # 股票收盘价、交易状态
    for data_name in ["S_DQ_ADJCLOSE","S_DQ_TRADESTATUS"]:
        print("个股数据:",data_name)
        data = model.get_specific_data(database = model.stock_database,
                                          collection = 'AShareEODPrices',
                                          start_date = start_date,
                                          end_date = end_date,
                                          date_name = 'TRADE_DT',
                                          stock_name = 'S_INFO_WINDCODE',
                                          target = data_name) 
        
        data = data.reindex(columns=stock_info.index.tolist())
        data.to_pickle("data/daily/stock/"+data_name)
    

        
    # 股票涨跌停状态、换手率、流通市值、总市值
    for data_name in ["UP_DOWN_LIMIT_STATUS","S_DQ_TURN"]:
        print("个股数据:",data_name)
        data = model.get_specific_data(database = model.stock_database,
                                          collection = 'AShareEODDerivativeIndicator',
                                          start_date = start_date,
                                          end_date = end_date,
                                          date_name = 'TRADE_DT',
                                          stock_name = 'S_INFO_WINDCODE',
                                          target = data_name) 
       
        data = data.reindex(columns=stock_info.index.tolist())
        data.to_pickle("data/daily/stock/"+data_name)

        
    # ST股票数据
    for data_name in ["ST_mark"]:
        print("个股数据:",data_name)
        data = model.get_specific_data(database = model.stock_database,
                                  collection = 'AShareST',
                                  start_date = start_date,
                                  end_date = end_date,
                                  date_name='date',           # 按照交易日（TRADE_DT）读取数据
                                  stock_name='S_INFO_WINDCODE',   # 股票代码
                                  target= data_name)            # 读取收盘价数据
        
        data = data.reindex(columns=stock_info.index.tolist())
        data.to_pickle("data/daily/stock/"+data_name)
        

# =============================================================================
# 计算日频交易日序列
# =============================================================================
    
    # 读取日频序列
    daily_dates_index = pd.read_pickle('data/daily/stock/S_DQ_ADJCLOSE').index
            
    # 生成Series
    daily_dates = pd.Series(daily_dates_index, index=daily_dates_index)
     
    # 数据存储
    daily_dates.to_pickle('data/basic/daily_dates')


# =============================================================================
# 计算上市日期
# =============================================================================
    
    # 计算股票上市日期
    idx = pd.read_pickle('data/basic/daily_dates')
    col = stock_info.index   
    
    # 展开上市日期序列
    list_date = pd.DataFrame(np.tile(stock_info.loc[:,"S_INFO_LISTDATE"],
                                      (idx.shape[0],1)),index=idx,columns=col)
        
    # 展开交易日序列         
    daily_date = pd.DataFrame(np.tile(idx,(col.shape[0],1)),index=col,columns=idx).T
    
    # 计算退市日期
    listed_days = daily_date-list_date
    
    # 数据存储
    listed_days.to_pickle('data/daily/stock/listed_days')

  

# =============================================================================
#  中国A股机构调研参与主体	AShareISParticipant
# =============================================================================

    print("机构调研主体")

    # 读取数据        
    cursor = model.other_database["AShareISParticipant"].find()
    
    # 读取数据, 存成DataFrame格式
    AShareISParticipant = model.cursor2dataframe(cursor, 10000)

              
# =============================================================================
#  中国A股机构调研活动	AshareISActivity
# =============================================================================

    print("机构调研活动")

    # 读取数据        
    cursor = model.other_database["AshareISActivity"].find()
    
    # 读取数据, 存成DataFrame格式
    AshareISActivity = model.cursor2dataframe(cursor, 10000)
    
    # 数据格式整理
    AshareISActivity = AshareISActivity[~AshareISActivity['S_SURVEYDATE'].isnull()]
    AshareISActivity['S_SURVEYDATE'] = [str(int(i)) for i in AshareISActivity['S_SURVEYDATE']]
    
    # 剔除部分只给出月份数的数据
    AshareISActivity = AshareISActivity[~(AshareISActivity['S_SURVEYDATE'].str.len()==6)]
    
    # 数据类型转换
    AshareISActivity['S_SURVEYDATE'] = pd.to_datetime(AshareISActivity['S_SURVEYDATE'])
    
# =============================================================================
#  AShareISParticipant和AshareISActivity数据预处理
# =============================================================================

    print("数据预处理")
    
    AshareISActivity_screen = AshareISActivity.loc[:,  
                ['EVENT_ID', 'S_INFO_WINDCODE', 'S_ACTIVITIESTYPE', 'S_SURVEYDATE', 'ANN_DT']]
    
    AShareISParticipan_screen = AShareISParticipant.loc[:, 
                ['EVENT_ID', 'S_INSTITUTIONNAME', 'S_INSTITUTIONTYPE']]
       
    # 数据拼接
    joined = pd.merge( AshareISActivity_screen, AShareISParticipan_screen,
                      how='inner', on=['EVENT_ID'])
    
    # 调取指数
    target = {'EVENT_ID':'调研活动代码',                                    
              'S_INFO_WINDCODE':'股票代码',                 
              'S_ACTIVITIESTYPE':'调研活动类型',
              'ANN_DT':'公告日期',
              'S_SURVEYDATE':'调查日期',
              'S_INSTITUTIONNAME':'调研机构名称',
              'S_INSTITUTIONTYPE':'调研机构类型'}

    # 列替换
    joined.rename(columns=target, inplace=True)
    
    # 投资者关系活动类别代码
    activities = {254001000:'特定对象调研', 254002000:'分析师会议',
                  254003000:'媒体采访', 254004000:'业绩说明会',
                  254005000:'新闻发布会', 254006000:'路演活动',
                  254007000:'现场参观', 254008000:'其他',
                  254009000:'投资者接待日活动', 254010000:'一对一沟通'}

    # 机构投资者类型
    institution = {257001001:'证券公司资管', 257001002:'证券公司自营',
                    257001003:'基金公司', 257001004:'保险公司',
                    257001005:'投资公司', 257001006:'外资机构',
                    257001007:'其他', 257002001:'证券公司', 
                    'nan':'nan'}
    
    # 数据替换
    joined['调研活动类型'] = joined['调研活动类型'].map(lambda x: activities[x])
    joined['调研机构类型'] = joined['调研机构类型'].map(
                    lambda x: (institution[x] if pd.notnull(x) else np.nan))
    
    # 机构归类
    institution = {'证券公司':'卖方', '证券公司资管':'公募',
                    '基金公司':'公募', '证券公司自营':'私募',
                    '投资公司':'私募', '外资机构':'外资',
                    '保险公司':'险资', '其他':'其他',  'nan':'nan'}
    
    # 归类后的映射关系
    joined['调研机构类型-归类'] = joined['调研机构类型'].map(
                    lambda x: (institution[x] if pd.notnull(x) else np.nan))
            
    # 日期替换
    joined['公告日期'] = pd.to_datetime(joined['公告日期'])
    joined.to_pickle('data/daily/other/surveydata')
        
    
# =============================================================================
# 统计机构调研活动后个股收益率
# =============================================================================


    # 读取数据
    survey = pd.read_pickle('data/daily/other/surveydata')
    
    # 去除非金融机构调研数据
    survey = survey[survey['调研机构类型'].notnull()]
    
    # 合并数据 
    survey_group = survey.groupby(['调研活动代码', '股票代码', 
            '调查日期', '公告日期']).count()['调研机构类型'].reset_index()
        
    
    # 剔除空值
    survey_group = survey_group.dropna(how='all')
    survey_group = survey_group.reset_index(drop=True)
      
    # 计算调查日期后N个交易日日期
    daily_dates = pd.read_pickle('data/basic/daily_dates')    
    
    # 计算调查日期之前的最近一个交易日以及后五个交易日
    days = 5
    survey_group['调查日期前最近交易日'] = survey_group['调查日期'].map(lambda x: daily_dates.loc[:x].iloc[-1])
    survey_group['调查日期后N个交易日'] = survey_group['调查日期前最近交易日'].map(
        lambda x: np.nan if x>daily_dates[-1-days] else daily_dates[(daily_dates<x).sum() + days])
    
    # 计算公告日期前最近交易日
    survey_group['公告日期前最近交易日'] = survey_group['公告日期'].map(lambda x: daily_dates.loc[:x].iloc[-1])
    survey_group.loc[survey_group['调查日期后N个交易日'].isnull(), '公告日期前最近交易日'] = np.nan
    
    # 两者取最新值
    survey_group['最终确定日期'] = survey_group.loc[:, ['公告日期前最近交易日','调查日期后N个交易日']].max(axis=1)
        
    # 五个交易日以后公布的调研数据不作考虑
    survey_group = survey_group[survey_group['公告日期'] <= survey_group['最终确定日期']]
    
    
    # 股票收盘价
    close_data = pd.read_pickle("data/daily/stock/S_DQ_ADJCLOSE")
    index_close = pd.read_pickle('data/daily/index/index_close')
    
    # 计算机构调研数据发布前收盘价数据
    survey_group['收益率'] = survey_group.apply(lambda x: close_data.loc[
        x['最终确定日期'], x['股票代码']] / close_data.loc[ x['调查日期前最近交易日'], x['股票代码']] - 1, axis=1)
    
        
    
    # 原始版本调研数据存储
    survey_group_1 = survey_group.groupby(['公告日期', '股票代码']).sum()
    stock_survey_pivot = survey_group_1.reset_index().pivot(
        index='公告日期', columns='股票代码', values='调研机构类型')
    
    # 按照日历日改写日期序列，填充空值
    stock_survey_pivot = stock_survey_pivot.resample('D').asfreq()
    
    # 行业数据        
    stock_survey_pivot = stock_survey_pivot.reindex(columns=stock_info.index)        
    
    # 调研次数空值替换为0
    stock_survey_pivot[stock_survey_pivot.isnull()] = 0
    
    # 数据存储
    stock_survey_pivot.to_pickle('data/daily/other/stock_survey')
            
    
