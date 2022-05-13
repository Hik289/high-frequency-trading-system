
import numpy as np
import numba
try:
    from numba import jitclass
except:
    from numba.experimental import jitclass
from numba import types, typed, njit
from numba import float64, int32

spec_backtestor = [('symbols', int32[:]),
                   ('hist_info', types.DictType(int32, float64[:, :])),
                   ('hist_idx', int32[:]),
                   ('start_time', float64),
                   ]


@jitclass(spec_backtestor)
class IntradayBacktestor(object):
    """
    每天实例化一个Backtestor对象
    @parameter:
        symbols: 股票代码, ["000001.SZ", ...]
        hist_info: 各股票tick数据 typed.Dict.empty(int32, float64[:, :])

    @member
        common
            hist_info 字典，键：股票索引
                            值：当日所有行情信息
            hist_idx 各股票tick行情推送的索引

    @method
        push 推送一支股票的一个tick
        run  模拟推送行情信息
    """

    def __init__(self, symbols, hist_info):
        self.symbols = symbols
        self.hist_info = hist_info
        self.hist_idx = np.full(len(symbols), 0, int32)

    def push(self, symbol, start_time):
        idx = self.hist_idx[symbol]
        t = self.hist_info[symbol][idx, 1]

        if np.floor(start_time) <= t < np.ceil(start_time):
            self.hist_idx[symbol] += 1
            return self.hist_info[symbol][idx, :]
        elif np.ceil(start_time) < t:
            #             print(start_time, t)
            return None

    def run(self, idh):
        start_time = 9 * 3600 - 3
        end_time = 15 * 3600 + 4
        time_delta = 0.5
        while start_time < end_time:
            for symbol in self.symbols:
                ticks = self.push(symbol, start_time)
                if ticks is not None:
                    print(symbol, self.hist_idx[symbol], ticks[1])
                    idh.new_tick(symbol, ticks)

            start_time += time_delta
        return idh
