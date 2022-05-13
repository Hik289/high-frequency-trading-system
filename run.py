from numba import typed, types, float64, int32
import numpy as np
import pandas as pd
import sys
from dqt import get_stock_taq, get_fut_taq, get_stock_daily, get_trading_day
from dqalpha import get_combined_index

from IntradayDataHandler import IntradayDataHandler
from Backtester import IntradayBacktestor
from dqt import DBUtils
dbu = DBUtils()


def trans(_tick):
    _tick['timestamp'] = _tick.datetime.astype(int).astype(float) / 1e9
    _tick['time'] = _tick['timestamp'] % 86400
    cols = ['timestamp',  'time',  'price',  'total_volume',  'value',
            's1',         's2',    's3',     's4',            's5',
            's1v',        's2v',   's3v',    's4v',           's5v',
            'b1',         'b2',    'b3',     'b4',            'b5',
            'b1v',        'b2v',   'b3v',    'b4v',           'b5v',
            ]
    return _tick[cols]


def save(data, file_name, key, date, symbol, path="./"):
    data = pd.DataFrame(data)
    data = data.dropna(how='all', axis=0)##.dropna(how='all', axis=1)
    
    data['date'] = date
    data['symbol'] = symbol
    data.to_hdf(path+file_name,
                key=key,
                append=True,
                mode='a',
                complevel=9,
                data_columns=None,
                complib='blosc:snappy',
                format='table',
                dropna=False)
    return


if __name__ == "__main__":
    date = sys.argv[1]  # "2021-02-10"
    trading_day = get_trading_day()
    last_5_tradingday = trading_day[trading_day['date']<date].tail(5).date.tolist()
    
    SYMBOLS = get_combined_index(start_date=date, end_date=date,
            indexes=['000016.SH', '000852.SH', '000300.SH', '000905.SH', '399006.SZ'],
            prefixes=[]).levels[1].tolist()
    # 剔除当日停牌股
    suspension_list = get_stock_daily(t='md', start_date=date, end_date=date, columns=['trade_status'])
    suspension_list = suspension_list[suspension_list.trade_status=='停牌'].index.droplevel(0).tolist()
    SYMBOLS = list(set(SYMBOLS) - set(suspension_list))
    # 剔除ST股
    ban_list = pd.read_sql_query(f"SELECT * FROM st_list WHERE date <=> '{date}'",con=dbu.stock_engine).symbol.tolist()
    SYMBOLS = list(set(SYMBOLS) - set(ban_list))
    #SYMBOLS = SYMBOLS[1500:]
    print('number of stocks:',len(SYMBOLS))
    #SYMBOLS = ['600053.SH','300628.SZ']

    # handler初始化参数
    all_daily = get_stock_daily('md',last_5_tradingday[0], date)
    OTHER_INFO = []
    SYMBOL_MAP = typed.Dict.empty(int32, types.string)
    up_limit = round(all_daily.loc[date].up_limit / all_daily.loc[date].factor, 2)
    down_limit = round(all_daily.loc[date].down_limit / all_daily.loc[date].factor, 2)
    preclose = round(all_daily.loc[last_5_tradingday[-1]].close / all_daily.loc[last_5_tradingday[-1]].factor, 2)
    average_5_days_amount = all_daily.loc[last_5_tradingday].groupby('symbol').turnover.mean()
    for i, symbol in enumerate(SYMBOLS):
        SYMBOL_MAP[int32(i)] = symbol
        OTHER_INFO.append([float64(i), 1.0 / len(SYMBOLS),up_limit.loc[date,symbol],down_limit.loc[date,symbol],preclose.loc[symbol],average_5_days_amount.loc[symbol]])
    OTHER_INFO = np.array(OTHER_INFO)
    # backtester初始化参数
    symbols = np.array(list(range(len(SYMBOLS))), dtype=np.int32)
    all_ticks = typed.Dict.empty(int32, float64[:, :])
    for i, symbol in enumerate(SYMBOLS):
        _tick = get_stock_taq(symbol, date, date)
        _tick = trans(_tick)
        all_ticks[int32(i)] = _tick.values
    bc = IntradayBacktestor(symbols=symbols, hist_info=all_ticks)
    
    idh = IntradayDataHandler(SYMBOL_MAP, int32(len(SYMBOLS)), OTHER_INFO)
    print('start running')
    idh = bc.run(idh)
    print('start saving')
    # save factor
    for i, symbol in enumerate(SYMBOLS):
        factor_bar_buffer = idh.factor_bar_buffer[i, :, :]#【哪只股票，bar index，哪个因子】
        save(data=factor_bar_buffer,
             file_name="./factor2/stock_intraday_factor_auction_{}.h5".format(date),
             key="factor_bar_buffer",
             date=date,
             symbol=symbol)


#         factor_tick_buffer = idh.factor_tick_buffer[i, :, :]
#         save(data=factor_tick_buffer,
#              file_name="./factor/stock_intraday_factor_{}.h5".format(date),
#              key="factor_tick_buffer",
#              date=date,
#              symbol=symbol)

#         mid_day_stat = idh.mid_day_stat[i, :].reshape(1, -1)
#         save(data=mid_day_stat,
#              file_name="./factor/stock_intraday_factor_{}.h5".format(date),
#              key="mid_day_stat",
#              date=date,
#              symbol=symbol)
