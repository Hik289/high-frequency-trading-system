import numpy as np
import time
from numba import (int32, float64,
                   typed, types,
                   )
try:
    from numba import jitclass
except:
    from numba.experimental import jitclass
from consts import handler_spec, BUFFER_SIZE_TICK
from dqalpha import calc_factor_ret
from dqalpha import neutralize, average_weighted, np_mean, np_sum
# 计算tick因子的函数
from factor_func import _calc_split_vol

# 计算截面因子（由tick）
from factor_func import(
    # tick level
    _calc_downVolatility,
    _calc_PVCorr,
    _calc_OutAmtRatio,
    _calc_RetBigOrder,
    _calc_BigBuyVolRatio,
    _calc_PriceWVolRatio,
    _calc_SkewWVol,
    _calc_AmountEntropy,
    _calc_PriceVolEntropy,
    # bar level
    _calc_relativePrice,
    _calc_turnoverAdj,
    # ts
    _calc_VolDivergent,
    _calc_ReverseSplit,
    _calc_priceBias,
    _calc_IVR,
    _calc_SpeVol,
    _calc_behaviorIndex,
    _calc_moment,
)


@jitclass(handler_spec)
class IntradayDataHandler(object):
    """
    日内逐笔 tick 数据，以及分钟bar数据

    @paramater：
        symbol_map: dict{int: string}, 股票代码索引对应字典
        symbol_size: int 股票个数
        other_info: float[:,:] 股票的其他信息，如昨日收盘价，指数权重等
        invoke_new_bar_seconds: int 触发bar的时间
        days_of_last: int 使用过去数据的天数

    @member:
        self.buffer_size_tick: int32, tick数据的缓存大小
        self.base_info_tick: list, tick数据基本信息
        self.tick_buffer: dict[string: float64[:,:]],
            tick数据缓存字典:
                键: string, 股票代码
                值: 二维数组, tick数据
        self.tick_index: dict,
            数据更新索引, 记录tick_buffer内的更新状态
                键: string, 股票代码
                值: int32, 索引值
    @method
        new_tick: 获取一只股票当前时刻行前的更新数据, 写入tick_buffer
        new_bar: 将分钟bar数据写入bar_buffer
        new_factor_tick: 将tick级别因子数据写入factor_tick_buffer
        new_factor_bar: 将bar级别因子数据写入factor_bar_buffer
        calc_tick_factor: 计算tick级别的因子
        update_bar: 定期将tick数据转化为分钟bar数据
            以及将factor_tick数据转化为factor_bar数据,
            并分别写入bar_buffer和factor_bar_buffer
        get_tick: 获取tick数据
        get_bar: 获取bar数据
        get_factor_tick: 获取factor_tick数据
        get_factro_bar: 获取factor_bar数据

    """

    def __init__(self,
                 symbol_map,  # symbol 映射列表   new_tick str -> int
                 symbol_size,
                 other_info,  # symbol, pre_close, std, float_value, ... , index_weight
                 invoke_new_bar_seconds=900,
                 days_of_last=5,
                 ):
        """

        @parameter：
            symbol_size: int, 股票个数
            index_weight: int, 指数的权重
            invoke_new_bar_seconds: 触发bar的时间
            days_of_last: 使用过去数据的天数
        """
        self.symbol_map = symbol_map
        # # 正则化指数权重
        self.other_info = other_info
        index_weight = self.other_info[:, 1]
        if np.sum(index_weight) != 1:
            index_weight = index_weight / np.sum(index_weight)

        self.symbol_size = symbol_size
        self.index_weight = np.full((symbol_size, 1), np.nan, float64)
        self.index_weight[:, 0] = index_weight
        self.invoke_new_bar_seconds = invoke_new_bar_seconds

        self.factor_tick_size = 6  # factor_tick 因子数
        self.factor_bar_size = 38  # factor_bar 因子数
        # 按日内描述确定
        # self.is_am = True # 初始状态 None 9:15-9:20 集合竞价_1 9:20-9:25 集合竞价_2 开盘段9:30-10:00 3 其他上午 4 下午收盘前:14:45 5 收盘段 6
        self.is_finished_last_bar = False

        self.days_of_last = days_of_last

        # tick数据
        # tick数据的表头
        self.buffer_size_tick = BUFFER_SIZE_TICK

        BASE_INFO_TICK = [
            "timestamp",    'time',     "price",    "volume",   'turnover',
            's1',           's2',       's3',       's4',       's5',
            's1v',          's2v',      's3v',      's4v',      's5v',
            'b1',           'b2',       'b3',       'b4',       'b5',
            'b1v',          'b2v',      'b3v',      'b4v',      'b5v',
            'limit'
        ]
        self.base_info_tick = typed.List.empty_list(types.string)
        for i in BASE_INFO_TICK:
            self.base_info_tick.append(i)
        # 初始化tick缓存
        self.tick_buffer = np.full((symbol_size,
                                    BUFFER_SIZE_TICK,
                                    len(BASE_INFO_TICK)),
                                   np.nan, float64)
        # 初始化tick索引
        self.tick_index = np.full(symbol_size, 0, int32)

        # bar数据
        # 分钟bar表头
        BUFFER_SIZE_BAR = 30
        self.buffer_size_bar = BUFFER_SIZE_BAR
        BASE_INFO_BAR = ['symbol_int',  'timestamp',    'open',     'high',
                         'low',         'close',        'volume',   'turnover',
                         'limit']
        self.base_info_bar = typed.List.empty_list(types.string)
        for i in BASE_INFO_BAR:
            self.base_info_bar.append(i)

        # 初始化bar缓存
        self.bar_buffer = np.full((symbol_size,
                                   BUFFER_SIZE_BAR,
                                   len(BASE_INFO_BAR)),
                                  np.nan, float64)
        # 初始化bar索引
        self.bar_index = 0

        # 初始化 factor_tick_buffer 数据
        self.factor_tick_buffer = np.full((symbol_size,
                                           BUFFER_SIZE_TICK,
                                           int32(self.factor_tick_size)),
                                          np.nan, np.float64)

        # 初始化 factor_bar 数据
        BASE_FACTOR_INFO = [
            'symbol_int',          'timestamp',

            'downVolality',        'PVCorr',              'OutAmtRatio',
            'RetBigOrder',         'BigBuyVolRatio',      'amount_each',
            'midPrice',            'dispersion',          'OIR',
            'SPREAD',              'SP_ret',              'entrust_diff_s1',
            'entrust_diff_b1',     'VOI',                 'MPB',
            'volatility',          'skew',

            'RPP',                  'turnoverAdj',        'ret_bar',
            'volume',

            'downVolality_RM',     'PVCorr_RM',           'OutAmtRatio_RM',
            'BigBuyVolRatio_RM',   'RPP_RM',              'turnoverAdj_RM',
            'VolDivergent',        'ReverseSplit',        'priceBias',
            'IVR',                 'SpecialVolatily',     'behaviorIndex',
            'retVolume',           'retVolatility',       'skewVolume',
        ]
        assert len(BASE_FACTOR_INFO) == self.factor_bar_size

        self.base_factor_info = typed.List.empty_list(types.string)
        for i in BASE_FACTOR_INFO:
            self.base_factor_info.append(i)

        # 初始化factor_bar_buffer数据
        self.factor_bar_buffer = np.full((symbol_size,
                                          BUFFER_SIZE_BAR,
                                          len(BASE_FACTOR_INFO)),
                                         np.nan, float64)

        # 初始化指数bar数据
        self.index_bar_buffer = np.full((BUFFER_SIZE_BAR,
                                         len(BASE_INFO_BAR)),
                                        np.nan, float64)
        self.index_bar_buffer[:, 0] = 0  # 指数代码
        # 初始化指数因子数据
        self.index_factor_buffer = np.full((BUFFER_SIZE_BAR,
                                            len(BASE_FACTOR_INFO)),
                                           np.nan, float64)
        self.index_factor_buffer[:, 0] = 0

        # 连续竞价对应的tick索引
        self.normal_tick_index = np.full((symbol_size, 2), np.nan, int32)
        self.normal_tick_index[:, 0] = -1
        self.normal_tick_index[:, 1] = BUFFER_SIZE_TICK

        # 上一个分钟bar索引
        self.last_bar_use_tick_index = np.full(symbol_size, -1, int32)

        # 一天的bar数据
        _lenBarOneDay = int32(np.ceil(14400 / invoke_new_bar_seconds))
        # 初始化过去一段时间的数据
        self.last_bar = np.full((symbol_size,
                                 days_of_last * _lenBarOneDay,
                                 len(BASE_INFO_BAR)),
                                np.nan, float64)
        self.last_factor = np.full((symbol_size,
                                    days_of_last * _lenBarOneDay,
                                    len(BASE_FACTOR_INFO)),
                                   np.nan, float64)
        self.last_other_data = np.full((days_of_last * _lenBarOneDay, 4),
                                       np.nan, float64)

        # 初始化 bar_data 和 factor_data
        self.bar_data = np.full((int32(symbol_size * BUFFER_SIZE_BAR),
                                 len(BASE_INFO_BAR)),
                                np.nan, float64)
        self.factor_data = np.full((int32(symbol_size * BUFFER_SIZE_BAR),
                                    len(BASE_FACTOR_INFO)),
                                   np.nan, float64)

        # 10点-15点的统计信息
        BASE_STAT_INFO = [
            'symbol_int',                   'timestamp',
            'retMean',      'retVar',       'retSkew',      'retKurt',
            'vwap_last15min',               'vwap_last3min',
        ]
        self.base_stat_info = typed.List.empty_list(types.string)
        for i in BASE_STAT_INFO:
            self.base_stat_info.append(i)

        self.ten_am_start_index = np.full(symbol_size, -1, int32)
        self.mid_day_stat = np.full((symbol_size, len(BASE_STAT_INFO)),
                                    np.nan, float64)
        # 集合竞价，连续竞价不同时间段的tick索引
        self.call_auction1 = np.full(
            (symbol_size, 2), -1, int32)  # 9:15-9:20
        self.call_auction2 = np.full(
            (symbol_size, 2), -1, int32)  # 9:20-9:25
        self.call_auction3 = np.full(
            (symbol_size, 2), -1, int32)  # 14:57-15:30
        self.continuous_auction1 = np.full(
            (symbol_size, 2), -1, int32)  # 9:30-10:00
        self.continuous_auction2 = np.full(
            (symbol_size, 2), -1, int32)  # 10:00-14:45
        self.continuous_auction3 = np.full(
            (symbol_size, 2), -1, int32)  # 14:45-14:57
        self.is_closed = np.full(symbol_size, 0, int32)

    def new_tick(self, symbol, tick):
        """
        获取一只股票当前时刻运行前的更新数据, 将其写入tick_buffer
        同时会计算factor_tick
        定期更新bar和factor_bar数据

        @parameter:
            symbol_int: int, 股票代码对应的整数映射
            tick: np.array(float64[:]) tick逐笔数据
                包括：
                timestamp: float64, 时间戳
                time: float64, 时间
                price: float64, 价格
                volume: float64, 累计成交量
                turnover: float64, 累计成交额
                s1, s2, s3, s4, s5: float64, 卖一至卖五价
                s1v, s2v, s3v, s4v, s5v: float64, 卖一至卖五量
                b1, b2, b3, b4, b5: float64, 买一至买五价
                b1v, b2v, b3v, b4v, b5v: float64, 买一至买五量

            tick传入后增加 limit: float64, 涨跌停状态
                    0: 正常， 1: 涨停，-1：跌停

            tick：
            ------
            0: timestamp	1: time    	    2: price   	    3: volume  	    4: turnover
            5: s1      	    6: s2      	    7: s3      	    8: s4      	    9: s5
            10: s1v     	11: s2v     	12: s3v     	13: s4v     	14: s5v
            15: b1      	16: b2      	17: b3      	18: b4      	19: b5
            20: b1v     	21: b2v     	22: b3v     	23: b4v     	24: b5v
            25: limit

        """
        assert len(self.base_info_tick) == len(tick) + 1

        symbol = int32(symbol)
        
        TIME = tick[1]  # 当前时间

        # 跳过非交易时间（盘后的最后一条tick保留）
        if not self._isTradeTime(TIME):
            return -1

        # 更新集合竞价，连续竞价不同时间段的tick索引
        self.update_auction_time(symbol, TIME)

        # # 1. 计算并更新tick因子
        self._update_tick_factor(symbol, tick)

        # 计算股票因子 （先计算因子再写入数据）
        self.update_factor(symbol, TIME)

        # 增加涨跌停状态
        limit = self._limit_state(symbol, tick)
        tick = np.array(list(tick) + [limit])
        
        # 存入tick_buff
        self.tick_buffer[symbol, self.tick_index[symbol]] = tick
        self.tick_index[symbol] += 1

        self._is_continue_bidding_time(symbol, TIME)
        
        if self.continuous_auction1[symbol, 1] > 0:
            return 1
            
        # 盘后更新的信息
        if TIME >= 54000 and not self.is_closed[symbol]:
            self._calc_infos_closed(symbol)
            self.is_closed[symbol] = 1
        return 0

    def update_factor(self, symbol, TIME):
        """
        (更新tick数据之前)
        计算股票因子
        步骤：
        1. 计算tick因子（放在new_tick函数里）
        2. 生成bar数据及指数bar
        3. 计算并更新截面bar因子
        4. 更新过去一段时间数据
        5. 计算并更新时间序列因子

        """
        symbol = int32(symbol)

        LAST_TIMESTAMP = self.tick_buffer[
            symbol, self.tick_index[symbol]-1, 0]
        if np.isnan(LAST_TIMESTAMP):
            return None

        # 判断是否触发bar
        if not self._is_invoke_bar(symbol, TIME):
            return None
        print("new bar")
        # 2. 更新bar数据
        self._update_bar(LAST_TIMESTAMP)
        # 更新指数bar数据
        self._update_index_bar(LAST_TIMESTAMP)

        # 3. 计算并更新截面因子
        self._update_section_factor(LAST_TIMESTAMP)

        # # 3.2 计算STAT
        # # 判断是否触发stat计算
        if self._is_invoke_stat(symbol, TIME):
            self._update_mid_day_stat(LAST_TIMESTAMP)

        # # 4.更新过去一段时间数据
        # self._update_last_data(LAST_TIMESTAMP)

        # # 5.计算并更新时间序列因子(主要是平均)
        # self._update_serise_factor()

        # 更新指数factor数据
        self._update_index_factor(LAST_TIMESTAMP)

        # 更新使用到的上一个tick索引
        self._update_last_tick_index()

        self._update_factor_data()

    def _isTradeTime(self, TIME):
        """
        判断是否为交易时间
        """
        if TIME < 33300 or TIME > 54003 or 41400 < TIME < 46800:
            return False
        else:
            return True

    def update_auction_time(self, symbol, TIME):
        """
        记录每个股票连续竞价开始和结束时的索引
        以二维数组的形式存储
        """
        # 9:15到来
        if TIME >= 33300 and self.call_auction1[symbol, 0] < 0:
            self.call_auction1[symbol, 0] = self.tick_index[symbol]
        # 9:20到来
        if TIME >= 33600 and self.call_auction2[symbol, 0] < 0:
            self.call_auction1[symbol, 1] = self.tick_index[symbol] - 1
            self.call_auction2[symbol, 0] = self.tick_index[symbol]
        # 9:25到来
        if TIME >= 33900 and self.call_auction2[symbol, 1] < 0:
            self.call_auction2[symbol, 1] = self.tick_index[symbol] - 1
        # 9:30到来
        if TIME >= 34200 and self.continuous_auction1[symbol, 0] < 0:
            self.continuous_auction1[symbol, 0] = self.tick_index[symbol]
        # 10:00到来
        if TIME >= 36000 and self.continuous_auction2[symbol, 0] < 0:
            self.continuous_auction1[symbol,
                                     1] = self.tick_index[symbol] - 1
            self.continuous_auction2[symbol, 0] = self.tick_index[symbol]
        # 14:45到来
        if TIME >= 53100 and self.continuous_auction3[symbol, 0] < 0:
            self.continuous_auction2[symbol,
                                     1] = self.tick_index[symbol] - 1
            self.continuous_auction3[symbol, 0] = self.tick_index[symbol]
        # 14:57到来
        if TIME >= 53820 and self.call_auction3[symbol, 0] < 0:
            self.continuous_auction3[symbol,
                                     1] = self.tick_index[symbol] - 1
            self.call_auction3[symbol, 0] = self.tick_index[symbol]
        return

    def _is_continue_bidding_time(self, symbol, TIME):
        """
        记录每个股票连续竞价开始和结束时的索引
        (写入tick数据之后判断)
        """

        # 记录连续竞价开始的索引
        if self.normal_tick_index[symbol, 0] == -1:
            if TIME >= 34200:
                self.normal_tick_index[symbol, 0] = self.tick_index[symbol] - 1
                return True
            return False
        # 记录连续竞价结束的索引
        else:
            # 1534723200 : "2018-08-20 "尾盘竞价开始实施的日期
            if TIME >= 53820:
                if self.normal_tick_index[symbol, 1] == self.buffer_size_tick:
                    self.normal_tick_index[symbol, 1] = self.tick_index[symbol]
                return False

            # 记录10点的索引
            if self.ten_am_start_index[symbol] == -1:
                if TIME >= 36000:
                    self.ten_am_start_index[symbol] = self.tick_index[symbol] - 1
            return True

    def _calc_infos_closed(self, symbol):
        # 最后15分钟vwap
        _start_index = self.continuous_auction3[symbol, 0]
        _end_index = self.tick_index[symbol] - 1
        _volume_15min = self.tick_buffer[symbol, _end_index, 3] - \
            self.tick_buffer[symbol, _start_index, 3]
        _amount_15min = self.tick_buffer[symbol, _end_index, 4] - \
            self.tick_buffer[symbol, _start_index, 4]
        vwap_15min = _amount_15min / _volume_15min / \
            100 if _volume_15min != 0 else np.nan
        # 最后3分钟vwap
        _start_index = self.call_auction3[symbol, 0]
        _end_index = self.tick_index[symbol] - 1
        _volume_3min = self.tick_buffer[symbol, _end_index, 3] - \
            self.tick_buffer[symbol, _start_index, 3]
        _amount_3min = self.tick_buffer[symbol, _end_index, 4] - \
            self.tick_buffer[symbol, _start_index, 4]
        vwap_3min = _amount_3min / _volume_3min / 100 if _volume_3min != 0 else np.nan
        self.mid_day_stat[symbol, 6:8] = np.array([vwap_15min, vwap_3min])
        return

    def _limit_state(self, symbol, tick):
        """
        判断该股票当前是否涨停或者跌停
        0：正常
        1：涨停
        -1：跌停
        """

        limit = 0.0
        if tick[5] == 0:  # s1 == 0
            limit = 1.0
        elif tick[15] == 0:  # b1 == 0
            limit == -1.0
        return limit

    def _update_tick_factor(self, symbol, tick):
        """
        计算并保存tick因子
        """
        symbol = int32(symbol)
        INDEX = self.tick_index[symbol]
        tick_factor = self._calc_tick_factor(symbol, tick)
        if tick_factor is not None:
            self.factor_tick_buffer[symbol, INDEX] = tick_factor

    def _is_invoke_stat(self, symbol, TIME):
        """
        是否触发计算日内统计信息（最后一个bar）
        """
        if not self._is_continue_bidding_time(symbol, TIME):
            if not self.is_finished_last_bar and TIME > 14 * 3600 + 57 * 60:
                self.is_finished_last_bar = True
                return True
        return False

    def _is_invoke_bar(self, symbol, TIME):
        """
        是否触发bar数据，以及是否更新bar因子的信号
        """
        symbol = int32(symbol)

        # 考虑最后一个bar生成
        if not self._is_continue_bidding_time(symbol, TIME):
            if not self.is_finished_last_bar and TIME > 14 * 3600 + 57 * 60:
                # self.is_finished_last_bar = True
                # 修改 is_finished_last_bar 留给 @_is_invoke_stat
                return True
            else:
                return False

        # 上一次更新bar的时间
        if self.last_bar_use_tick_index[symbol] == -1:  # 第一次触发bar
            _last_time = max(9.5 * 3600, self.tick_buffer[symbol, 0, 1])
        else:
            _last_time = np.ceil(
                self.tick_buffer[
                    symbol,
                    self.last_bar_use_tick_index[symbol],
                    1] / self.invoke_new_bar_seconds
            ) * self.invoke_new_bar_seconds

        # 当前时间是否超过触发更新时间
        if TIME - _last_time > self.invoke_new_bar_seconds:
            # 上午跳到下午的情况
            if TIME - _last_time > 5000:
                if TIME - _last_time > 5400 + self.invoke_new_bar_seconds:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def _update_bar(self, TIMESTAMP):
        """
        更新bar数据

        @parameter：
            TIMESTAMP: float64, 时间
        """

        INDEX = self.bar_index

        # 时间
        self.bar_buffer[:, INDEX, 1] = TIMESTAMP

        for symbol in range(self.symbol_size):
            self.bar_buffer[symbol, INDEX, 0] = symbol
            bar = self._calc_bar(symbol)
            self.bar_buffer[symbol, INDEX, 2:] = bar

        # 保存为2维数据
        self.bar_data[INDEX*self.symbol_size:(INDEX+1)
                      * self.symbol_size] = self.bar_buffer[:, INDEX]

        self.bar_index += 1

    def _calc_bar(self, symbol):
        """
        计算bar数据
        @parameter：
            symbol: int32, 股票代码

            tick: np.array(float64[:]) tick逐笔数据
            -------
            0: timestamp	1: time    	    2: price   	    3: volume  	    4: turnover
            5: s1      	    6: s2      	    7: s3      	    8: s4      	    9: s5
            10: s1v     	11: s2v     	12: s3v     	13: s4v     	14: s5v
            15: b1      	16: b2      	17: b3      	18: b4      	19: b5
            20: b1v     	21: b2v     	22: b3v     	23: b4v     	24: b5v
            25: limit

            bar: float64[:]
            ---------
            0: timestamp  1: time       2: open       3: high       4: low
            5: close      6: volume     7: turnover   8: limit

        """
        symbol = int32(symbol)

        # 初始化bar数据
        bar = np.full(int32(len(self.base_info_bar)-2),
                      np.nan, float64)  # 只计算除时间部分bar数据

        # 选取这段时间的tick数据
        if self.last_bar_use_tick_index[symbol] == -1:  # 第一次计算bar数据
            tick_buffer = self.tick_buffer[
                symbol, self.normal_tick_index[symbol, 0]: self.tick_index[symbol]]
        else:
            tick_buffer = self.tick_buffer[
                symbol, self.last_bar_use_tick_index[symbol] + 1: self.tick_index[symbol]]

        # 无tick数据
        if len(tick_buffer) == 0:
            return bar

        tick_price = tick_buffer[:, 2]
        tick_volume = tick_buffer[:, 3]
        tick_turnover = tick_buffer[:, 4]
        tick_limit = tick_buffer[:, -1]

        # 将tick数据转化为分钟该数据
        bar[0] = tick_price[0]      # open
        bar[1] = max(tick_price)    # high
        bar[2] = min(tick_price)    # low
        bar[3] = tick_price[-1]     # close
        bar[4] = tick_volume[-1]    # volume
        bar[5] = tick_turnover[-1]  # turnover
        bar[6] = tick_limit[-1]     # limit

        return bar

    def _calc_tick_factor(self, symbol, tick):
        """
        计算tick数据因子factor_tick
        (在写入tick数据之前先计算tick因子)

        @parameter：
            symbol: int, 股票代码对应的整数映射

            tick: np.array(float64[:]) tick逐笔数据
            -------
            0: timestamp	1: time    	    2: price   	    3: volume  	    4: turnover
            5: s1      	    6: s2      	    7: s3      	    8: s4      	    9: s5
            10: s1v     	11: s2v     	12: s3v     	13: s4v     	14: s5v
            15: b1      	16: b2      	17: b3      	18: b4      	19: b5
            20: b1v     	21: b2v     	22: b3v     	23: b4v     	24: b5v
            25: limit

        @return：
            factor_tick: None or np.array(float64[:]),
                当时间处于集合竞价时，不计算factor_tick返回None
                否则，根据tick数据计算因子，返回factor_tick

            factor_tick (factor_1):
            ------------
            0:  mid_price 	    1:  ret_tick         2: dispersion       3: OIR
            4:  SPREAD        	5:  SP_ret           6: entrust_diff_s1  7: entrust_diff_b1
            8:  VOI	            9:  MPB             10: volume_buy      11: volume_sell
            12: amount_buy      13: amount_sell

            """
        symbol = int32(symbol)
        # 取上个tick数据
        if self.tick_index[symbol] == 1:
            tick_last = tick
            volume = tick[3]
            amount = tick[4]
        else:
            tick_last = self.tick_buffer[symbol, self.tick_index[symbol]-1]
            volume = tick[3] - tick_last[3]
            amount = tick[4] - tick_last[4]

        price = tick[2]
        price_last = tick_last[2]
        Pb = tick[15:20]
        Ps = tick[5:10]
        Vb = tick[20:25]
        Vs = tick[10:15]
        Vb_last = tick_last[20:25]
        Vs_last = tick_last[10:15]

        # 计算TICK因子
        # 中间价
        mid_price = (tick[5] + tick[15]) / 2  # s1 + b1 / 2
        # 收益率
        ret_tick = tick[2] / tick_last[2] - 1 if tick_last[2] != 0 else np.nan
        

        # 分拆买卖的成交量和成交额
        volume_buy, volume_sell, amount_buy, amount_sell \
            = _calc_split_vol(amount=amount,
                              volume=volume,
                              s1_last=tick_last[5],
                              b1_last=tick_last[15])

        factor_tick = np.array([
            mid_price,       ret_tick,
            volume_buy,      volume_sell,    amount_buy,     amount_sell
        ])

        return factor_tick

    def _update_section_factor(self, TIMESTAMP):
        """
        计算并更新截面因子
        @parameter:

            bar: float64[:]
            ---------
            0: symbol_int 1: timestamp  2: open       3: high       4: low
            5: close      6: volume     7: turnover   8: limit

            (factor_1)
                factor_bar: float64[:,:,:] (factor_1)
                    ------
                    0: symbol_int           1: timestamp

                    #  factor_by_tick
                    2:  downVolality        3: PVCorr               4: OutAmtRatio
                    5:  RetBigOrder         6: BigBuyVolRatio       7: amount_each
                    8:  midPrice            9: dispersion          10: OIR
                    11: SPREAD             12: SP_ret              13: entrust_diff_s1
                    14: entrust_diff_b1    15: VOI                 16: MPB
                    17: volatility,        18: skew

                    # factor_by_bar
                    19: RPP                20: turnoverAdj         21: ret_bar
                    22: volume    

                    # series_factor
                    23: downVolality_RM     24: PVCorr_RM           25: OutAmtRatio_RM
                    26: BigBuyVolRatio_RM   27: RPP_RM              28: turnoverAdj_RM
                    29: VolDivergent        30: ReverseSplit        31: priceBias
                    32: IVR                 33: SpecialVolatily     34: behaviorIndex
                    35: retVolume           36: retVolatility       37: skewVolume
            (factor_2)
                factor_bar: float64[:,:,:] (factor_1)
                    ------
                    0: symbol_int           1: timestamp

                    #  factor_by_tick
                    2: PriceWVolRatio        3: SkewWVol        4: AmountEntropy
                    5: HighTrade             6: PriceVolEntropy
        """
        INDEX = self.bar_index - 1
        # 保存数据
        self.factor_bar_buffer[:, INDEX, 1] = TIMESTAMP

        # 由 tick 数据计算
        for symbol in range(self.symbol_size):
            self.factor_bar_buffer[symbol, INDEX, 0] = symbol
            factor_by_tick = self._calc_factor_by_tick(symbol)
            # 保存数据
            self.factor_bar_buffer[symbol, INDEX, 2:7] = factor_by_tick

        # # 由 bar 数据计算
        factor_by_bar = self._calc_factor_by_bar()
        # # 保存数据
        self.factor_bar_buffer[:, INDEX, 19:22] = factor_by_bar

    def _calc_factor_by_tick(self, symbol):
        """
        由tick_factor和tick计算bar因子

        @parameter

            tick: np.array(float64[:]) tick逐笔数据
            -------
            0: timestamp	1: time    	    2: price   	    3: volume  	    4: turnover
            5: s1      	    6: s2      	    7: s3      	    8: s4      	    9: s5
            10: s1v     	11: s2v     	12: s3v     	13: s4v     	14: s5v
            15: b1      	16: b2      	17: b3      	18: b4      	19: b5
            20: b1v     	21: b2v     	22: b3v     	23: b4v     	24: b5v
            25: limit

            (factor_1)
                factor_tick (factor_1):
                ------------
                0:  mid_price 	    1:  ret_tick         2: volume_buy      3: volume_sell
                4: amount_buy      5: amount_sell

        @return
            (factor_1)
                factor_by_tick(factor_1)
                --------
                0:  downVolality        1: PVCorr              2:  OutAmtRatio
                3:  RetBigOrder         4: BigBuyVolRatio      5:  amount_each
                6:  midPrice            7: dispersion          8:  OIR
                9:  SPREAD             10: SP_ret              11: entrust_diff_s1
                12: entrust_diff_b1    13: VOI                 14: MPB
                15: volatility,        16: skew

            factor_2: 
                factor_by_tick:
                    0: PriceWVolRatio        1: SkewWVol        2: AmountEntropy
                    3: HighTrade             4: PriceVolEntropy
        """
        symbol = int32(symbol)

        # 选取这段时间的tick_factor数据
        _start_index = max(self.last_bar_use_tick_index[symbol] + 1,
                           self.normal_tick_index[symbol, 0])
        _end_index = self.tick_index[symbol]
        # 数据准备
        # ret_tick = factor_tick_buffer[:, 1]
        price = self.tick_buffer[symbol, _start_index:_end_index, 2]
        volume = self.tick_buffer[symbol, _start_index:_end_index, 3]
        amount = self.tick_buffer[symbol, _start_index:_end_index, 4]
        if _start_index > 0:
            volume -= self.tick_buffer[symbol, _start_index-1:_end_index-1, 3]
            amount -= self.tick_buffer[symbol, _start_index-1:_end_index-1, 4]
        else:
            volume[1:] -= volume[:-1]
            amount[1:] -= volume[:-1]

        # factor_2
        PriceWVolRatio = _calc_PriceWVolRatio(price, volume)
        SkewWVol = _calc_SkewWVol(price, volume)
        AmountEntropy = _calc_AmountEntropy(amount)
        HighTrade = np.mean(
            np.array([PriceWVolRatio,  SkewWVol, AmountEntropy]))
        PriceVolEntropy = _calc_PriceVolEntropy(price, volume)

        factor_by_tick = np.array([
            PriceWVolRatio,     SkewWVol,             AmountEntropy,
            HighTrade,          PriceVolEntropy,
        ])
        return factor_by_tick

    def _calc_factor_by_bar(self,):
        """
        由 bar 数据计算

        @parameter:

            bar: float64[:]
            ---------
            0: timestamp  1: time       2: open       3: high       4: low
            5: close      6: volume     7: turnover   8: limit

            daily_factor: float[:,:]
            -----------
            0: factor       1: close        2: turnover
            3: pb           4: trad_share

        @return
            factor_by_bar:
            --------------
            0: RPP      1: turnoverAdj      3: ret_bar
            4: volume    
        """

        INDEX = self.bar_index - 1  # 当前bar数据索引
        # factor_1
        price = self.bar_buffer[:, INDEX, 5]
        low = self.bar_buffer[:, INDEX, 4]
        high = self.bar_buffer[:, INDEX, 3]
        amount = self.bar_buffer[:, INDEX, 7]
        volume = self.bar_buffer[:, INDEX, 6]  # 成交量
        if INDEX > 0:
            amount -= self.bar_buffer[:, INDEX-1, 7]
            volume -= self.bar_buffer[:, INDEX - 1, 6]
        # mkt_value = price * self.daily_factor[:, 4] \
        #     * self.daily_factor[:, 0] * 100  # close * share * div_fac

        # 相对价格位置
        RPP = _calc_relativePrice(price, high, low)
        # # 市值调整后的换手率
        # turnoverAdj = _calc_turnoverAdj(mkt_value, amount)

        # bar收益率
        if INDEX == 0:
            ret_bar = self.bar_buffer[:, INDEX, 5] \
                / self.bar_buffer[:, INDEX, 2] - 1
        else:
            ret_bar = self.bar_buffer[:, INDEX, 5] \
                / self.bar_buffer[:, INDEX - 1, 5] - 1
        factor_by_bar = np.stack((RPP, ret_bar, volume), axis=1)

        return factor_by_bar

    # def _update_last_data(self, TIMESTAMP):
    #     """
    #     更新过去一段时间的数据缓存

    #     @parameter:
    #         last_bar: float64[:,:,:]
    #             -----------------------
    #             0: symbol_int 1: timestamp  2: open       3: high       4: low
    #             5: close      6: volume     7: turnover   8: limit

    #         daily_factor: float[:,:]
    #             -----------
    #             0: factor       1: close        2: turnover
    #             3: pb           4: trad_share

    #         last_factor: float64[:,:,:] 过去N天的factor数据
    #         --------------------
    #             0: symbol_int           1: timestamp

    #             #  factor_by_tick
    #             2:  downVolality        3: PVCorr               4: OutAmtRatio
    #             5:  RetBigOrder         6: BigBuyVolRatio       7: amount_each
    #             8:  midPrice            9: dispersion          10: OIR
    #             11: SPREAD             12: SP_ret              13: entrust_diff_s1
    #             14: entrust_diff_b1    15: VOI                 16: MPB
    #             17: volatility,        18: skew

    #             # factor_by_bar
    #             19: RPP                20: turnoverAdj         21: ret_bar
    #             22: volume

    #             # serise_factor
    #             23: downVolality_RM     24: PVCorr_RM           25: OutAmtRatio_RM
    #             26: BigBuyVolRatio_RM   27: RPP_RM              28: turnoverAdj_RM
    #             29: VolDivergent        30: ReverseSplit        31: priceBias
    #             32: IVR                 33: SpecialVolatily     34: behaviorIndex
    #             35: retVolume           36: retVolatility       37: skewVolume

    #      """

    #     INDEX = self.bar_index-1
    #     # 更新last_bar数据
    #     self.last_bar[:, :-1] = self.last_bar[:, 1:]
    #     self.last_bar[:, -1] = self.bar_buffer[:, INDEX]
    #     # 价格数据乘复权因子
    #     self.last_bar[:, -1, 2:6] *= self.daily_factor[:, 0:1]

    #     # 更新last_factor 数据
    #     self.last_factor[:, :-1] = self.last_factor[:, 1:]
    #     self.last_factor[:, -1] = self.factor_bar_buffer[:, INDEX]

    # def _update_serise_factor(self):
    #     """
    #     计算时间序列因子数据
    #     @parameter:
    #         factor_1:
    #             section_factor: float64[:,:] 新计算的截面因子数据
    #                 ------
    #                 0: downVolality     1: PVCorr           2: OutAmtRatio
    #                 3: RetBigOrder      4: BigBuyVolRatio   5: RPP

    #             daily_factor: float[:,:]
    #                 -----------
    #                 0: close        1: factor       2: turnover
    #                 3: pb           4: trad_share   5：industry

    #             last_bar: float64[:,:,:]
    #                 ---------
    #                 0: symbol_int 1: timestamp  2: open       3: high       4: low
    #                 5: close      6: volume     7: turnover   8: limit

    #             last_factor: float64[:,:,:] 过去N天的factor数据
    #                 ------
    #                 0: symbol_int           1: timestamp

    #                 #  factor_by_tick
    #                 2:  downVolality        3: PVCorr               4: OutAmtRatio
    #                 5:  RetBigOrder         6: BigBuyVolRatio       7: amount_each
    #                 8:  midPrice            9: dispersion          10: OIR
    #                 11: SPREAD             12: SP_ret              13: entrust_diff_s1
    #                 14: entrust_diff_b1    15: VOI                 16: MPB
    #                 17: volatility,        18: skew

    #                 # factor_by_bar
    #                 19: RPP                20: turnoverAdj         21: ret_bar
    #                 22: volume

    #                 # serise_factor
    #                 23: downVolality_RM     24: PVCorr_RM           25: OutAmtRatio_RM
    #                 26: BigBuyVolRatio_RM   27: RPP_RM              28: turnoverAdj_RM
    #                 29: VolDivergent        30: ReverseSplit        31: priceBias
    #                 32: IVR                 33: SpecialVolatily     34: behaviorIndex
    #                 35: retVolume           36: retVolatility       37: skewVolume
    #     """

    #     INDEX = self.bar_index - 1
    #     # factor_1
    #     volume = self.last_factor[:, :, 22]
    #     price = self.last_bar[:, :, 5]
    #     ret_bar = self.last_factor[:, :, 21]
    #     amount_each = self.last_factor[:, :, 7]
    #     fama_french = self.last_other_data[:, 1:]  # MKT SMB HML

    #     # 对因子求平均
    #     downVolality_RM = np_mean(self.last_factor[:, :, 2], axis=1)
    #     PVCorr_RM = np_mean(self.last_factor[:, :, 3], axis=1)
    #     OutAmtRatio_RM = np_mean(self.last_factor[:, :, 4], axis=1)
    #     BigBuyVolRatio_RM = np_mean(self.last_factor[:, :, 6], axis=1)
    #     RPP_RM = np_mean(self.last_factor[:, :, 18], axis=1)
    #     turnoverAdj_RM = np_mean(self.last_factor[:, :, 19], axis=1)
    #     # 成交量分歧度
    #     VolDivergent = _calc_VolDivergent(volume)
    #     # 反转拆分
    #     ReverseSplit = _calc_ReverseSplit(amount_each)
    #     # 价格偏离度
    #     priceBias = _calc_priceBias(price, ret_bar, N=10)
    #     # 特异度
    #     _len_fama = min(ret_bar.shape[1], len(fama_french))
    #     IVR = _calc_IVR(ret_bar[:, :_len_fama], fama_french[:_len_fama])
    #     # 特异性波动率
    #     SpecialVolatily = _calc_SpeVol(ret_bar[:, :_len_fama],
    #                                    fama_french[:_len_fama])
    #     # 交易热度
    #     behaviorIndex = _calc_behaviorIndex(IVR, turnoverAdj_RM, priceBias)

    #     serise_factor = np.stack((
    #         downVolality_RM,        PVCorr_RM,          OutAmtRatio_RM,
    #         BigBuyVolRatio_RM,      RPP_RM,             turnoverAdj_RM,
    #         VolDivergent,           ReverseSplit,       priceBias,
    #         IVR,                    SpecialVolatily,    behaviorIndex
    #     ), axis=1)

    #     INDEX = self.bar_index - 1
    #     self.factor_bar_buffer[:, INDEX, 23:35] = serise_factor

    #     # factor_2:
    #     serise_factor = np.full((int32(self.symbol_size), 3),
    #                             np.nan, float64)
    #     for symbol in range(self.symbol_size):
    #         ret_each = self.last_factor[symbol, :, 21]
    #         volume_each = self.last_factor[symbol, :, 22]
    #         volume_each[volume_each == 0] = np.nan
    #         volatility = self.last_factor[symbol, :, 17]
    #         volatility[volatility == 0] = np.nan
    #         skew = self.last_factor[symbol, :, 18]

    #         retVolume = average_weighted(ret_each, 1 / volume_each) \
    #             - average_weighted(ret_each,  volume_each)
    #         retVolatility = average_weighted(ret_each, 1 / volatility) \
    #             - average_weighted(ret_each,  volatility)
    #         skewVolume = average_weighted(skew, 1 / volume_each) \
    #             - average_weighted(skew,  volume_each)
    #         serise_factor[symbol, -3:] = [retVolume, retVolatility, skewVolume]

    #     # 按行业标准化
    #     industry = self.daily_factor[:, 5]
    #     for i in range(serise_factor.shape[1]):
    #         serise_factor[:, i] = neutralize(serise_factor[:, i], industry)

    #     self.factor_bar_buffer[:, INDEX, 35:] = serise_factor

    def _update_index_bar(self, TIMESTAMP):
        """
        更新指数bar数据
        """
        INDEX = self.bar_index - 1
        self.index_bar_buffer[INDEX, 1] = TIMESTAMP
        self.index_bar_buffer[INDEX, 2:] = np_sum(
            self.bar_buffer[:, INDEX, 2:] * self.index_weight, axis=0)

    def _update_index_factor(self, TIMESTAMP):
        """
        更新指数factor数据
        """
        INDEX = self.bar_index - 1
        self.index_factor_buffer[INDEX, 1] = TIMESTAMP
        self.index_factor_buffer[INDEX, 2:] = np_sum(
            self.factor_bar_buffer[:, INDEX, 2:] * self.index_weight, axis=0)

    def _update_last_tick_index(self):
        """
        更新使用到的上一个tick索引
        """
        for symbol in range(self.symbol_size):
            self.last_bar_use_tick_index[symbol] = self.tick_index[symbol] - 1

    def _update_factor_data(self):
        """
        更新2维的因子数据
        """
        INDEX = self.bar_index - 1
        self.factor_data[INDEX*self.symbol_size: (
            INDEX+1)*self.symbol_size] = self.factor_bar_buffer[:, INDEX]

    def _update_mid_day_stat(self, TIMESTAMP):
        """
        更新日内统计数据
        """
        for symbol in range(self.symbol_size):
            self.mid_day_stat[symbol, 0] = symbol
            self.mid_day_stat[symbol, 1] = TIMESTAMP
            mid_day_stat = self._calc_mid_day_stat(symbol)
            self.mid_day_stat[symbol, 2:6] = mid_day_stat

    def _calc_mid_day_stat(self, symbol):
        """
        计算stat数据

        @parameter

            tick: np.array(float64[:]) tick逐笔数据
            -------
            0: timestamp	1: time    	    2: price   	    3: volume  	    4: turnover
            5: s1      	    6: s2      	    7: s3      	    8: s4      	    9: s5
            10: s1v     	11: s2v     	12: s3v     	13: s4v     	14: s5v
            15: b1      	16: b2      	17: b3      	18: b4      	19: b5
            20: b1v     	21: b2v     	22: b3v     	23: b4v     	24: b5v
            25: limit

            factor_tick:
            ------------
            0:  mid_price 	    1:  ret_tick         2: dispersion       3: OIR
            4:  SPREAD        	5:  SP_ret           6: entrust_diff_s1  7: entrust_diff_b1
            8:  VOI	            9:  MPB             10: volume_buy      11: volume_sell
            12: amount_buy      13: amount_sell

        @retutn 
            mid_day_stat: float64[:]
            ---------------------
            0:  retMean,       1: retVar,       2:  retSkew,            3: retKurt,

        """
        symbol = int32(symbol)
        mid_day_stat = np.full(4, np.nan, float64)
        # 选取这段时间的tick_factor数据
        _start_index = max(self.ten_am_start_index[symbol], 0)
        _end_index = self.tick_index[symbol]
        if _start_index >= _end_index:
            return mid_day_stat
        TICK_BUFFER = self.tick_buffer[symbol, _start_index:_end_index]
        FACTOR_TICK_BUFFER = self.factor_tick_buffer[symbol, _start_index: _end_index]

        # 计算统计信息
        ret_tick = FACTOR_TICK_BUFFER[:, 1]
        amount_buy = FACTOR_TICK_BUFFER[:, 12]
        amount_sell = FACTOR_TICK_BUFFER[:, 13]
        vol = TICK_BUFFER[:, 3]

        # 矩信息
        retMean, retVar, retSkew, retKurt = _calc_moment(ret_tick)
        annu_tick_factor = 252 * 4 * 60 * 20
        retMean *= annu_tick_factor
        retVar *= annu_tick_factor

        mid_day_stat = np.array([retMean,       retVar,     retSkew,    retKurt,
                                 ])
        return mid_day_stat

    def get_bar(self):
        """

        获取bar数据
        """
        return self.bar_data[:self.bar_index*self.symbol_size]

    def get_factor(self):
        """
        获取factor_bar数据
        """
        return self.factor_data[:self.bar_index*self.symbol_size]

    def get_index_factor(self):
        """
        获取指数的bar因子
        """
        return self.index_factor_buffer[:self.bar_index]

    def get_index_bar(self):
        """
        获取指数的bar数据
        """
        return self.index_bar_buffer[:self.bar_index]

    def get_other_data(self):
        """
        获取更新过后的其他的数据（此时为fama三因子数据）
        """
        return self.last_other_data[-self.bar_index:]

    def add_daily_factor(self, daily_factor):
        """
        传入每个股票当日的基本数据
        0： factor (除权因子)
        """
        self.daily_factor = daily_factor

    def add_last_bar(self, last_bar):
        """
        输入过去 N 日的 bar 数据 （价格数据已经除权）

        bar: float64[:]
        ---------
        0: symbol_int 1: timestamp  2: open       3: high       4: low
        5: close      6: volume     7: turnover   8: limit
        """
        _len = min(self.last_bar.shape[1], last_bar.shape[1])
        self.last_bar[:, -_len:] = last_bar[:, -_len:]

    def add_last_factor(self, last_factor):
        """
        输入过去 N 日的 factor 数据
        """
        _len = min(self.last_factor.shape[1], last_factor.shape[1])
        self.last_factor[:, -_len:] = last_factor[:, -_len:]

    def add_last_other_data(self, last_ohter_data):
        """
        """
        _len = min(self.last_other_data.shape[0], last_ohter_data.shape[0])
        self.last_other_data[-_len:] = last_ohter_data[-_len:]