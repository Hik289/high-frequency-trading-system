from functools import wraps
import os
import logging
import time
import numpy as np
import pandas as pd
from numba import jit, njit
import arrow
from tables.exceptions import HDF5ExtError
from dqt import get_stock_daily, get_trading_day
from dqalpha import np_corr, calc_factor_ret, isna_2d


default_start_date = '2018-01-01'
CALENDAR = get_trading_day().index


def _trans_timestamp(timestamp, unit):
    """
    将时间转化为给定unit的时间，比如：unit=60 表示将时间转化为分钟为单位
    """

    res = np.ceil(timestamp / unit) * unit
    return res


def utc_to_unix(utc_time):
    """
    将utc时间转化为unix时间格式

    @parameter：
        utc_time：datetime or str, utc时间

    @return：
        unix_time: float，unix时间
    """
    return float(arrow.get(utc_time).timestamp)


def unix_to_date(unix_time):
    """
    将unix时间转化为datetime格式的日期
    """

    return arrow.get(unix_time).date()


def repeat_func(ERROR, logging_info=None, n_try=5, sleep_time=1):
    """
    用于重复一个函数的装饰器

    @parameter：
        ERROR：error, 错误类型
        logging_info: str, 提示信息
        n_try: int, 重复次数
        sleep_time: int, 失败后休息的时间
    """
    def repeat_func_decorator(function):
        @wraps(function)
        def decorated(*args, **kwargs):
            nonlocal n_try
            connected = False
            res = None
            while not connected:
                try:
                    res = function(*args, **kwargs)
                    connected = True
                except ERROR:
                    if n_try == 0:
                        raise ERROR
                    n_try -= 1
                    logging.warning(logging_info)
                    time.sleep(sleep_time)
            return res
        return decorated
    return repeat_func_decorator


@repeat_func(HDF5ExtError, '请关闭h5文件', 5, 30)
def read_h5(key,
            file_path=None,
            fileName=None,
            start_date='2016-01-04',
            end_date=None,
            symbols=None,
            columns=None,
            base_dir='./',
            **kwargs):
    """
    读取H5文件数据
    """
    if file_path is None:
        if fileName is None:
            raise "请输入 file_path 或者 fileName 至少一个"
        else:
            file_path = os.path.join(base_dir, fileName)

    _start = utc_to_unix(start_date)
    if kwargs.get('where'):
        kwargs['where'] += ' & timestamp >= _start'
    else:
        kwargs['where'] = 'timestamp >= _start'

    if end_date:
        _end = utc_to_unix(end_date) + 24*3600
        kwargs['where'] += ' & timestamp < _end'

    if symbols:
        kwargs['where'] += ' & symbol = symbols'

    if columns:
        if isinstance(columns, str):
            columns = [columns]
        kwargs['where'] += ' & columns in columns'

    with pd.HDFStore(file_path) as store:
        if key in store:
            df = store.select(key, **kwargs)
            return df
    return None


def get_history_factor(key='factor_1',
                       start_date='2016-01-04',
                       end_date=None,
                       symbols=None,
                       columns=None,
                       file_path=None,
                       factorFileName='factor_bar.h5',
                       base_dir='./',
                       **kwargs):
    """
    获取历史的因子数据
    @parameter:
    """

    return read_h5(key=key,
                   file_path=file_path,
                   fileName=factorFileName,
                   start_date=start_date,
                   end_date=end_date,
                   symbols=symbols,
                   columns=columns,
                   base_dir=base_dir,
                   **kwargs)


@repeat_func(HDF5ExtError, '请关闭h5文件', 5, 30)
def get_history_bar(key='bar',
                    start_date='2016-01-04',
                    end_date=None,
                    symbols=None,
                    columns=None,
                    file_path=None,
                    barFileName='bar_50.h5',
                    base_dir='./',
                    **kwargs):
    """
    获取历史的bar数据
    """

    _start = utc_to_unix(start_date)
    if kwargs.get('where'):
        kwargs['where'] += ' & timestamp >= _start'
    else:
        kwargs['where'] = 'timestamp >= _start'

    if end_date is not None:
        _end = utc_to_unix(end_date) + 24 * 3600
        kwargs['where'] += ' & timestamp < _end'

    if symbols is not None:
        kwargs['where'] += 'symbol = symbols'

    if columns:
        if isinstance(columns, str):
            columns = [columns]
        kwargs['where'] += ' & columns in columns'

    if file_path is None:
        if barFileName is None:
            raise "请输入 file_path 或者 barFileName 至少一个"
        else:
            file_path = os.path.join(base_dir, barFileName)
    with pd.HDFStore(file_path) as store:
        df = store.select(key, **kwargs)
        return df


def read_min_ic(ic_key='factor_1',
                ic_fileName='ic_bar_50.h5',
                ic_filePath=None,
                invoke_new_bar_seconds=900):
    """
    读取因子在每个bar上的IC值
    """
    ic_df = read_h5(key=ic_key, file_path=ic_filePath, fileName=ic_fileName)
    ic_df.index = _trans_timestamp(ic_df['timestamp'], 900) % (24*3600)
    ic_df.drop('timestamp', inplace=True, axis=1)
    ic_min_df = ic_df.groupby('timestamp'
                              ).apply(lambda x: pd.DataFrame(
                                  {'IC': x.mean(),
                                   'ICIR': x.mean() / x.std()}
                              )).unstack().loc[9.5*3600:15*3600]
    ic_min_df.index = pd.to_datetime(ic_min_df.index, unit='s').time
    return ic_min_df


def yiled_chunk_date(start_date, end_date, chunk_size=50):
    """
    划分时间区间
    """
    date_range = CALENDAR.loc[start_date:end_date]
    if len(date_range) < chunk_size:
        yield date_range[0].date(), date_range[-1].date()
    i = chunk_size
    while i <= len(date_range):
        yield date_range[i-chunk_size].date(), date_range[i-1].date()
        i += chunk_size
    if i - chunk_size < len(date_range):
        yield date_range[i-chunk_size].date(), date_range[-1].date()


@njit
def calc_min_ic(arr):
    """
    计算分钟因子的ic值：

    @parameter：
        arr: np.array(float64[:,:]) 时间、分钟因子，下期收益率（当前分钟收盘价于下一日开盘价的收益率）
            表头为：['timestamp', (因子)factor1, factot2,..., (下期收益率)'ret_next']
    @return:
        ic_arr: np.array(float64[:,:]), 每分钟的ic值
        表头为: ['timestamp', (因子)factor1, factot2,...]

    """
    t_lst = np.unique(arr[:, 0])
    ic_arr = np.full((len(t_lst), arr.shape[1]-1), np.nan)
    ic_arr[:, 0] = t_lst
    for i in range(len(t_lst)):
        t = t_lst[i]
        for j in range(1, arr.shape[1]-1):
            ic_arr[i, j] = np_corr(arr[arr[:, 0] == t][:, j],
                                   arr[arr[:, 0] == t][:, -1])
    return ic_arr


@njit
def add_const(X):
    """
    增加常数项
    """
    col_size = X.shape[1] + 1
    res = np.full((len(X), col_size), np.nan)
    res[:, 0] = 1
    res[:, 1:] = X
    return res


@njit
def calc_reg_param(X, y):
    """
    计算回归的参数

    X: float64[:,:]
        shape[0]: 数据长度
        shape[1]: 指标数目
    """

    _not_na_index = ~(isna_2d(X) | np.isnan(y))
    X_not_na = X.copy()
    y_not_na = y.copy()
    X_not_na = X_not_na[_not_na_index]
    y_not_na = y_not_na[_not_na_index]
    if len(X_not_na) == 0:
        return np.full(1, np.nan)
    # OLS回归参数
    param = np.dot(np.dot(np.linalg.pinv(np.dot(X_not_na.T,
                                                X_not_na)),
                          X_not_na.T),
                   y_not_na)
    return param


@njit
def calc_reg_fit_value(X, y, is_add_const=True):
    """计算回归的拟合值"""

    if is_add_const:
        X = add_const(X)
    param = calc_reg_param(X, y)
    if np.any(np.isnan(param)):
        return np.full(len(y), np.nan)
    y_hat = np.dot(X, param)
    return y_hat


@njit
def calc_resid(X, y, is_add_const=True):
    """计算回归的残差"""

    y_hat = calc_reg_fit_value(X, y, is_add_const=is_add_const)
    resid = y - y_hat
    return resid


@njit
def calc_r_square(X, y, is_add_const=True):
    """
    计算回归的拟合优度 R^2

    @parater:
        X: float[:, :]
        y: float[:]
    """

    if np.all(np.isnan(X)) or np.all(np.isnan(y)) or np.all(y == 0):
        return np.nan
    y_bar = np.mean(y)
    y_hat = calc_reg_fit_value(X, y, is_add_const=is_add_const)

    SST = np.nansum(np.power(y - y_bar, 2))
    SSR = np.nansum(np.power(y_hat - y_bar, 2))
    if SST != 0:
        r_square = SSR / SST
    else:
        r_square = np.nan
    return r_square


def _calc_factor_ret(*args):
    return


@njit
def _calc_fama_french_bar(merge_arr):
    """
    计算fama-french三因子的bar数据
    """
    bar_time = np.sort(np.unique(merge_arr[:, 0]))
    factor_ret = [([bar_time[i],  # timestamp
                    np.nansum(merge_arr[merge_arr[:, 0] == bar_time[i], 3]
                              * merge_arr[merge_arr[:, 0] == bar_time[i], 4]),  # MKT
                    calc_factor_ret(merge_arr[merge_arr[:, 0] == bar_time[i], 1],
                                    merge_arr[merge_arr[:, 0] == bar_time[i], 3]),  # SMB
                    calc_factor_ret(merge_arr[merge_arr[:, 0] == bar_time[i], 2],
                                    merge_arr[merge_arr[:, 0] == bar_time[i], 3])])  # HML
                  for i in range(len(bar_time))
                  ]
    return factor_ret


@repeat_func(HDF5ExtError, '请关闭h5文件', 5, 30)
def get_fama_french(start_date, end_date,
                    index_kind='sz50',
                    invoke_second=900,
                    fileName='fama_french.h5',
                    base_dir='./',):
    """
    获取fama-french三因子的bar数据
    """

    file_path = os.path.join(base_dir, fileName)

    # 尝试提取数据
    _start = utc_to_unix(start_date)
    _end = utc_to_unix(end_date) + 24*3600
    try:
        with pd.HDFStore(file_path) as store:
            last_timestamp = max(store.select_column(index_kind, 'timestamp'))
            if last_timestamp > _end - 24 * 3600:
                return store.select(index_kind,
                                    where='timestamp >= _start & timestamp < _end')
            else:
                _start_date_update = unix_to_date(last_timestamp + 24*3600)
    except KeyError:
        _start_date_update = pd.to_datetime('2016-01-04').date()

    _end_date_update = (pd.to_datetime('today') - pd.Timedelta(days=1)).date()

    # 提取数据失败，开始更新数据
    if index_kind == 'sz50':
        barFileName = 'bar_50.h5'
        symbol_int_key = 'sz50'
    elif index_kind == 'zz500':
        barFileName = 'bar_500.h5'
        symbol_int_key = 'zz500'
    else:
        raise ValueError("get_fama_french：index_kind请输入sz50或者zz500")

    # 分钟bar数据
    bar = get_history_bar(start_date=_start_date_update,
                          end_date=_end_date_update,
                          barFileName=barFileName,
                          columns=['timestamp', 'open',
                                   'close', 'symbol'])
    bar['date'] = pd.to_datetime(bar['timestamp'], unit='s').dt.normalize()
    bar['ret'] = bar['close'] / bar['open'] - 1
    bar['bar_time'] = _trans_timestamp(bar['timestamp'], invoke_second)

    # 日数据
    market_df = get_stock_daily(start_date=_start_date_update,
                                end_date=_end_date_update,
                                t='merged',
                                columns=['date', 'symbol', 'pb',
                                         'trad_share', 'close',
                                         'factor'])
    market_df.rename(columns={'close': 'close_day',
                              'pb': 'pb_day'}, inplace=True)
    # 权重数据
    symbol_int_map = get_history_bar(start_date=_start_date_update,
                                     end_date=_end_date_update,
                                     barFileName='symbol_int_map.h5',
                                     key=symbol_int_key)
    symbol_int_map['weight'] /= 100

    # 数据合并
    merged_df = bar.merge(market_df, on=['date', 'symbol'], how='left'
                          ).merge(symbol_int_map[['symbol', 'weight']],
                                  on=['date', 'symbol'], how='left')
    merged_df['value'] = merged_df['close'] * merged_df['trad_share']
    merged_df['pb'] = merged_df['pb_day'] / merged_df['close_day'] * \
        merged_df['close'] * merged_df['factor']

    # 计算因子收益
    merged_arr = merged_df[['bar_time', 'pb', 'value', 'ret', 'weight']].values
    factor_ret = _calc_fama_french_bar(merged_arr)
    fama_french = pd.DataFrame(
        factor_ret, columns=['timestamp', 'MKT', 'SMB', 'HML'])

    # 保存数据
    with pd.HDFStore(file_path) as store:
        store.append(index_kind, fama_french,
                     format='t', data_columns=True)
        return store.select(index_kind,
                            where='timestamp >= _start & timestamp < _end')


@njit
def _calc_future_bar_ret(bar_time, close_arr, shift=1):
    """
    计算期货合约的bar收益率
    @bar_time: float[:], bar时间序列
    @close_arr: float[:, :], 2d, 时间和收盘价
    @shift: int, 收益率滞后期数

    return
    @ret_arr: float[:], len(ret_arr) = len(bar_time) - 2*shift
    """
    ret_arr = np.full(len(bar_time)-shift*2, np.nan)
    for i in range(shift, len(bar_time)-shift):
        _close = close_arr[(close_arr[:, 0] > bar_time[i-1])
                           & (close_arr[:, 0] <= bar_time[i]), 1]
        if len(_close) == 0 or _close[0] == 0:
            continue
        ret_arr[i-shift] = _close[-1] / _close[0] - 1
    return ret_arr


@njit
def _calc_future_ic(factor_arr):
    """
    计算期货主力合约的IC值
    @factor_arr: float[:,:]
        0: timestamp, 
        1...-1: factor, 
        -1: ret
    """
    timestamp = _trans_timestamp(factor_arr[:, 0], 24*3600)
    dates = np.unique(timestamp)
    ret_arr = np.full((len(dates), factor_arr.shape[1] - 1), np.nan)
    for i in range(len(dates)):
        _factor = factor_arr[timestamp == dates[i]]
        ret_arr[i, 0] = dates[i]
        if len(_factor) < 8:
            continue
        for j in range(1, factor_arr.shape[1]-1):
            ret_arr[i, j] = np.corrcoef(_factor[:, j], _factor[:, -1])[0, 1]
    return ret_arr


def calc_future_ic(variety='IH',
                   factor_key='factor_1',
                   start_date='2016-01-04',
                   end_date=None,
                   shift=1):
    """
    计算期货主力合约的因子IC值
    """

    if start_date is None:
        start_date = '2016-01-04'
    if end_date is None:
        end_date = pd.to_datetime('today').date()
    # 获得指数因子值
    fileNameMap = {'IH': 'index_factor_bar_50.h5'}
    # TODO 增加其他两个指数

    factor_df = read_h5(key=factor_key,
                        fileName=fileNameMap[variety],
                        start_date=start_date,
                        end_date=end_date)
    if factor_df is None or len(factor_df) == 0:
        raise ValueError("无因子数据")

    factor_df.drop_duplicates(inplace=True)

    # 选取有因子的时间
    _start, _end = factor_df['timestamp'].agg(['min', 'max'])
    start_date = unix_to_date(_start)
    end_date = unix_to_date(_end)

    # 获得每日主力合约
    dbu = DBUtils()
    main_future_df = dbu.get_main_daily(variety=variety,
                                        start_date=start_date,
                                        end_date=end_date)

    main_future_df['date'] = main_future_df['date'].shift(
        1)  # 前移一天，因为主力切换第二天才知道。
    main_date_info = main_future_df.groupby('symbol'
                                            ).agg({'date': ['first', 'last']}
                                                  ).date
    # 计算相关系数
    corr_lst = []
    for symbol, (first, last) in main_date_info.iterrows():
        # 选取因子数据
        _factor_df = factor_df.loc[lambda x: (x['timestamp'] >= utc_to_unix(first))
                                   & (x['timestamp'] < utc_to_unix(last) + 24*3600)].copy()
        bar_time = _factor_df.timestamp.values

        # 读取期货逐笔数据
        tick_df = get_fut_taq(symbol, first, last)
        if tick_df.shape[0] == 0:
            continue

        # 转化成UTS
        tick_df['timestamp'] = tick_df['datetime'].astype(int) // 10**9
        close_arr = tick_df[['timestamp', 'last']].values
        # 计算收益率
        _factor_df.loc[lambda x: x['timestamp'].isin(bar_time[:-shift*2]),
                       'ret_next'] = _calc_future_bar_ret(bar_time,
                                                          close_arr,
                                                          shift=shift)
        # 计算相关系数
        factor_arr = _factor_df.iloc[shift:-shift].values
        ic_arr = _calc_future_ic(factor_arr)

        # 保存数据
        corr_lst.append(ic_arr)

        del tick_df, _factor_df

    ic_df = pd.DataFrame(np.concatenate(corr_lst),
                         columns=factor_df.columns)

    del factor_df
    return ic_df


def _calc_quantile(factor, _quantiles=5):
    try:
        return pd.qcut(factor, _quantiles, labels=False, duplicates="drop") + 1
    except Exception as e:
        raise e
