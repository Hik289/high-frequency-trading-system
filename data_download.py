# -*- coding: utf-8 -*-

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

        # 获取一致预取数据库对象
        self.est_database = self.client["xquant_est"]
        
        # 获取指数数据库对象
        self.index_database = self.client["xquant_index"]
        
        
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
                             date_name, stock_name, target):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        if collection in ['AShareBalanceSheet','AShareCashFlow','AShareIncome']:
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}, 'STATEMENT_TYPE':408001000},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
        else:
            
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            
        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 100000)
        
        # 业绩预告数据容易出现重复
        if collection in ['AShareProfitNotice',"HKIndexEODPrices","HKStockHSIndustriesMembers","AShareMoneyFlow"]:
            
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
    
    

if __name__ == "__main__":
    
    model = data_model()
    
    start_date = "2000-01-01"
    end_date = "2021-09-10"


    # # =============================================================================
    # # 读取股票列表
    # # =============================================================================
    
    # # 最新A股信息
    # stock_info = pd.DataFrame(list(model.stock_database["AShareDescription"].find({}, {'_id':0})))
    # stock_info = stock_info.set_index('S_INFO_WINDCODE')
    
    # # 剔除A和T开头的股票代码
    # stock_info = stock_info.loc[[not i.startswith('A') for i in stock_info.index], :]
    # stock_info = stock_info.loc[[not i.startswith('T') for i in stock_info.index], :]
    
    # # 股票代码顺序重置
    # stock_info.sort_index(inplace=True)
    # stock_info.to_pickle('data/stock_info')


    # # =============================================================================
    # # 个股数据：后复权收盘价、成交量、成交额
    # # =============================================================================

    # stock_dict = {"S_DQ_ADJCLOSE":"stock_close",
    #               "S_DQ_VOLUME":"stock_volume",
    #               "S_DQ_AMOUNT":"stock_amount"}

    # for idx in stock_dict.keys():
        
    #     print(idx)
        
    #     stock_data = model.get_specific_data(database = model.stock_database,
    #                                           collection = "AShareEODPrices",
    #                                           start_date = start_date,
    #                                           end_date = end_date,
    #                                           date_name = "TRADE_DT",
    #                                           stock_name = "S_INFO_WINDCODE",
    #                                           target = idx)
        
    #     # 数据存储
    #     stock_data = stock_data.reindex(columns=stock_info.index)
    #     stock_data.to_pickle("data/"+stock_dict[idx])


    # # =============================================================================
    # # 个股数据：换手率、涨跌停情况、流通股本
    # # =============================================================================

    # stock_dict = {"S_DQ_TURN":"stock_turn",
    #               "UP_DOWN_LIMIT_STATUS":"updown_limit",
    #               "FLOAT_A_SHR_TODAY":"float_share"}
    
    # for idx in stock_dict.keys():
        
    #     print(idx)
        
    #     stock_data = model.get_specific_data(database = model.stock_database,
    #                                           collection = "AShareEODDerivativeIndicator",
    #                                           start_date = start_date,
    #                                           end_date = end_date,
    #                                           date_name = "TRADE_DT",
    #                                           stock_name = "S_INFO_WINDCODE",
    #                                           target = idx)
        
    #     # 数据存储
    #     stock_data = stock_data.reindex(columns=stock_info.index)
    #     stock_data.to_pickle("data/"+stock_dict[idx])


    # # =============================================================================
    # # wind行业数据
    # # =============================================================================
    
    # # # 中国A股Wind行业指数日行情	AIndexWindIndustriesEOD    
    # # # 中国A股万得指数成份股	AIndexMembersWIND
        
    # # 行业收益率
    # indus_data = model.get_specific_data(database = model.indus_database,
    #                                       collection = "AIndexWindIndustriesEOD",
    #                                       start_date = start_date,
    #                                       end_date = end_date,
    #                                       date_name = "TRADE_DT",
    #                                       stock_name = "S_INFO_WINDCODE",
    #                                       target = "S_DQ_CLOSE")
    
    # index = pd.read_excel('data/指数类型.xlsx', index_col=0)
    # indus_data = indus_data.loc[:, index['证券代码']].reindex(index=stock_data.index)

    # # 行业指数收盘价
    # indus_data.to_pickle('data/Wind_indus_close')
    
    # # -------------------------------------------------------------------------
    # # 成分股进出数据（只包含成分股什么时候进入什么时候退出）
    # # -------------------------------------------------------------------------         
    # # 指数成分股数据
    # IndexMember = model.get_all_data(database=model.indus_database, 
    #                             collection='AIndexMembersWIND',  # 中国A股指数成份股
    #                             start_date=start_date,
    #                             end_date=end_date, 
    #                             date_name='S_CON_INDATE')  # 按照数据日期读取数据
  
    # # 日期替换
    # IndexMember['S_CON_INDATE'] = pd.to_datetime(IndexMember['S_CON_INDATE'])
    # IndexMember['S_CON_OUTDATE'] = pd.to_datetime(IndexMember['S_CON_OUTDATE'])
    # IndexMember = IndexMember.loc[:, ['F_INFO_WINDCODE', 'S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE']]
    
    # # 数据名称整理
    # IndexMember.rename(columns={'F_INFO_WINDCODE':'指数代码', 'S_CON_WINDCODE':'股票代码',
    #                             'S_CON_INDATE':'纳入日期', 'S_CON_OUTDATE':'剔除日期'},inplace=True)
        
    # # 提取目标
    # index = pd.read_excel('data/指数类型.xlsx', index_col=0)
    # IndexMember = IndexMember[IndexMember['指数代码'].isin(index['证券代码'])]

    # # 指数成分股数据存储
    # IndexMember.to_pickle('data/Wind_indus_member')



    # =============================================================================
    # 将成分股进出记录转换为股票行业归属（待修正）
    # =============================================================================
    
    index = pd.read_excel('data/指数类型.xlsx', index_col=0)
    IndexMember = pd.read_pickle('data/Wind_indus_member')
    
           
    # =============================================================================
    # 数据预处理程序：计算行业换手率（待修正）
    # =============================================================================

    # 读取数据文件
    stock_share = pd.read_pickle("data/float_share")
    indus_belong = pd.read_pickle("data/indus_belong")
    stock_volume = pd.read_pickle("data/stock_volume")
    stock_amount = pd.read_pickle("data/stock_amount")
    stock_close = pd.read_pickle("data/stock_close")
    
    # 日频交易日序列
    daily_dates = stock_close.index
    
    target_indus = pd.read_excel('data/指数类型.xlsx', index_col=0)['证券代码'].tolist()

    index_volume = pd.DataFrame(index=daily_dates, columns=target_indus)
    index_amount = pd.DataFrame(index=daily_dates, columns=target_indus)
    index_turn = pd.DataFrame(index=daily_dates, columns=target_indus)
    
    for indus in target_indus:
        index_volume[indus] = stock_volume[indus_belong == indus].sum(axis=1)
        index_amount[indus] = stock_amount[indus_belong == indus].sum(axis=1)
        index_turn[indus] = stock_volume[indus_belong == indus].sum(axis=1) / \
                            stock_share[indus_belong == indus].sum(axis=1)
    # 条件储存
    index_volume.loc['2005-01-01':,:].to_pickle("data/daily/indus/index_volume")
    index_amount.loc['2005-01-01':,:].to_pickle("data/daily/indus/index_amount")
    index_turn.loc['2005-01-01':,:].to_pickle("data/daily/indus/index_turn")

    
    