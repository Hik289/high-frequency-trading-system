try:
    from numba import jitclass
except:
    from numba.experimental import jitclass
from numba import typed, typeof, types, boolean
from numba import int32, float64, boolean, int64
import numpy as np
from numba import vectorize, njit
from utils import calc_resid, calc_r_square, _calc_factor_ret
from dqalpha import np_mean, np_percentile, calc_quantile, calc_entropy, average_weighted

WEIGHT_VOLUME = np.arange(1, 0, -0.2,)


"""
======================
1. 计算 TICK 因子的函数
=====================
"""


@njit
def _calc_sale_split_factor(amount, volume, s1_last, b1_last):
    """
    计算将买卖成交量/成交额拆分的因子 r
    """
    if volume:
        price_avg = amount / volume / 100
    else:
        return np.nan

    if price_avg >= s1_last:
        r = 1
    elif price_avg <= b1_last:
        r = 0
    else:
        if s1_last - b1_last:
            r = (price_avg - b1_last) / (s1_last - b1_last)
        else:
            r = np.nan
    return r


@njit
def _calc_split_vol(amount, volume, s1_last, b1_last):
    """
    计算分拆买卖的成交量和成交额
    """
    r = _calc_sale_split_factor(amount=amount,
                                volume=volume,
                                s1_last=s1_last,
                                b1_last=b1_last)
    volume_buy = r * volume
    volume_sell = (1 - r) * volume
    amount_buy = r * amount
    amount_sell = (1 - r) * amount
    return volume_buy, volume_sell, amount_buy, amount_sell


"""
===========================================
2. 计算截面 BAR 因子的函数——由 TICK 因子计算得到
===========================================
"""


@njit
def _calc_downVolatility(ret):
    """
    [TICK]
    计算下行波动率

    @ret: float[:]
    """
    if ret.size == 0 or np.all(np.isnan(ret)):
        return np.nan
    if np.nansum(ret**2) == 0:
        return np.nan
    downVolality = (np.nansum(ret[ret < 0]**2) / np.nansum(ret**2))**0.5

    return downVolality


@njit
def _calc_PVCorr(prcie, volume):
    """
    计算量价相关性
    price: tick数据
    volume : tick级别数据
    """
    if np.nansum(volume) == 0:
        return np.nan
    volume_norm = volume / np.nansum(volume)
    PVCorr = np.corrcoef(prcie, volume_norm)[0, 1]

    return PVCorr


@njit
def _calc_OutAmtRatio(amount_cum, ret):
    """
    单笔流出金额占比
    """
    _ret = ret[1:]
    amount = amount_cum[1:] - amount_cum[:-1]
    if np.mean(amount) != 0:
        OutAmtRatio = np.mean(amount[_ret < 0]) / np.mean(amount)
    else:
        OutAmtRatio = np.nan
    return OutAmtRatio


@njit
def _calc_RetBigOrder(amount_cum, ret, thershold=0.7):
    """
    大单推动涨跌幅
    """

    amount = amount_cum[1:] - amount_cum[:-1]
    _ret = ret[1:]
    index_big_order = amount > np.nanpercentile(amount, thershold * 100)
    RetBigOrder = np.nanprod(1 + _ret[index_big_order]) - 1

    return RetBigOrder


@njit
def _calc_BigBuyVolRatio(amount_cum, amount_buy):
    """
    计算大买成交额占比
    """
    all_amount = amount_cum[-1] - amount_cum[0]
    if not all_amount:
        return np.nan

    big_amount_buy = np.nansum(amount_buy[
        amount_buy > np.mean(amount_buy) + np.nanstd(amount_buy)])

    BigBuyVolRatio = big_amount_buy / all_amount

    return BigBuyVolRatio


@njit
def _calc_moment(ret):
    """
    计算收益率的1,2,3,4阶矩
    """
    if len(ret) == 0 or np.all(np.isnan(ret)):
        return np.full(4, np.nan)
    n = np.sum(~np.isnan(ret))
    retMean = np.mean(ret)
    retVar = np.nanvar(ret)
    if retVar == 0:
        retVar = np.nan
    retSkew = 1 / n * np.nansum((ret - retMean)**3) / retVar**(3/2)
    retKurt = 1 / n * np.nansum((ret - retMean)**4) / retVar**2 - 3
    return np.array([retMean, retVar, retSkew, retKurt])


@njit
def _calc_PriceWVolRatio(price, vol):
    """
    计算 加权收盘价比
    """
    price_mean = np.mean(price)
    if price_mean == 0:
        return np.nan
    price_weight_vol = average_weighted(price, vol)
    PriceWVolRatio = price_weight_vol / price_mean
    return PriceWVolRatio


@njit
def _calc_SkewWVol(price, vol):
    """
    计算 加权偏度
    """
    price_std = np.nanstd(price)
    if price_std == 0:
        return np.nan
    price_hat = ((price - np.mean(price)) / price_std)**3
    SkewWVol = average_weighted(price_hat, vol)
    return SkewWVol


@njit
def _calc_AmountEntropy(amount):
    """
    计算 成交额熵
    """
    amount_all = np.nansum(amount)
    if amount_all == 0:
        return np.nan
    amount_ratio = amount / amount_all
    AmountEntropy = calc_entropy(amount_ratio)
    return AmountEntropy


@njit
def _calc_PriceVolEntropy(price, vol):
    """
    计算 单位成交额占比熵
    """
    vol_sum = np.nansum(vol)
    if vol_sum == 0:
        return np.nan
    price_sum = np.nansum(price)
    if price_sum == 0:
        return np.nan
    price_vol = price / price_sum * vol / vol_sum
    PriceVolEntropy = calc_entropy(price_vol)
    return PriceVolEntropy


"""
===========================================
3. 计算截面 BAR 因子的函数——由 BAR 因子计算得到
===========================================
"""


@njit
def _calc_relativePrice(price, high, low):
    """
    计算相对价格位置：
    TODO 记得滚动平均

    @price: float[:]
    @high: float[:]
    @low: float[:]

    @RPP: float[:]
    """
    high_minu_low = high - low
    high_minu_low[high_minu_low == 0] = np.nan
    RPP = (price - low) / (high - low)

    return RPP


@njit
def _calc_turnoverAdj(mkt_value, amount):
    """
    计算市值调整后的换手率：换手对市值回归后的残差
    """
    turnover_adj = np.full(len(mkt_value), np.nan)
    mkt_value[mkt_value == 0] = np.nan
    amount[amount == 0] == np.nan
    if np.all(np.isnan(mkt_value)):
        return turnover_adj
    turnoverRate = amount / mkt_value  # 换手率
    X = np.log(mkt_value).reshape((-1, 1))
    y = np.log(turnoverRate)

    turnover_adj = calc_resid(X, y, is_add_const=True)
    return turnover_adj


"""
===========================================
3. 计算过去一段时间的 BAR 因子的函数
===========================================
"""


@njit
def _calc_VolDivergent(volume):
    """
    计算成交量分歧

    @parameter:
        volume : float[:,:] 过去N日的bar数据

    @return:
        VolDivergent: float[:]
    """
    _vol_diff = (np_percentile(volume, 0.9*100, axis=1)
                 - np_percentile(volume, 0.1*100, axis=1))
    _index_nonzero = _vol_diff != 0
    VolDivergent = np_percentile(volume, 0.5*100, axis=1)
    VolDivergent[_index_nonzero] /= _vol_diff[_index_nonzero]
    VolDivergent[~_index_nonzero] = 0
    return VolDivergent


@njit
def _calc_ReverseSplit(amount_each):
    """
    计算拆分反转

    """
    ReverseSplit = np_percentile(amount_each, 70, axis=1) \
        - np_percentile(amount_each, 30, axis=1)

    return ReverseSplit


@njit
def _calc_priceBias(price, ret, N=10):
    """
    计算价差偏离度
    price: float64[:,:] 过去一段时间所有股票的所有bar价格
    ret: float64[:,:]
    """
    print(price, ret)
    priceSpread = np.full(price.shape, np.nan)
    priceBias = np.full(len(price), np.nan)
    dist = 1 - np.corrcoef(ret)  # 两个股票的距离
    for symbol in range(len(ret)):
        rf_price = np_mean(price[np.argsort(dist[symbol])[:N]],
                           axis=0)  # 距离最近的N个股票的平均价格
        rf_price[rf_price == 0] = np.nan
        priceSpread[symbol] = np.log(price[symbol]) \
            - np.log(rf_price)
        _std = np.nanstd(priceSpread)
        if _std != 0:
            priceBias[symbol] = (priceSpread[symbol, -1]
                                 - np.mean(priceSpread)) / _std
    return priceBias


@njit
def _calc_IVR(ret, fama_french):
    """
    计算特异度：1 - 收益率对三因子的回归R方
    """
    IVR = np.full(len(ret), np.nan)
    for symbol in range(len(ret)):
        IVR[symbol] = 1 - calc_r_square(fama_french,
                                        ret[symbol],
                                        is_add_const=True)
    return IVR


@njit
def _calc_SpeVol(ret, fama_french):
    """
    计算特异性波动率： 收益率对三因子残差的波动率
    """
    SpeVol = np.full(len(ret), np.nan)
    for symbol in range(len(ret)):
        SpeVol[symbol] = np.nanstd(calc_resid(fama_french,
                                              ret[symbol],
                                              is_add_const=True))
    return SpeVol


@njit
def _calc_behaviorIndex(IVR, turnoverAdj, priceBias):
    """
    计算交易热度
    """
    behaviorIndex = (calc_quantile(IVR)
                     + calc_quantile(turnoverAdj)
                     + calc_quantile(priceBias)) / 3
    return behaviorIndex