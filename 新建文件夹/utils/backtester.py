import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from utils import kiwi_operators as kwo
from utils import config
from utils import data_environment as env
from .feature_selection import basic
from .feature_selection import bootstrap
from importlib import reload

# reload(kwo)
# reload(config)


def calc_sharpe(feat: pd.DataFrame, cost: float = 4e-4):
    pos = kwo.calc_pos(feat)
    pos = pos.rolling(config.fwd_day).mean()
    cur_fwd = env.fwd_one.reindex(pos.index)
    se_ret = (pos * cur_fwd).sum(axis=1)
    mv_pos = pos - pos.shift()
    mv_pos = mv_pos.abs()
    se_cost = cost * mv_pos.sum(axis=1)
    se_ret = se_ret - se_cost
    return kwo.calc_sharpe_from_ret(se_ret)


def get_metrics(feat: pd.DataFrame, cost: float = 4e-4):
    pos_raw = kwo.calc_pos(feat)
    pos_one = pos_raw.rolling(config.fwd_day).mean()
    fwd_raw = env.fwd_ret.copy().reindex(pos_raw.index)
    fwd_one = env.fwd_one.copy().reindex(pos_one.index)

    res_dt = dict()

    # time_series
    res_dt["ic"] = kwo.calc_ic(pos_raw, fwd_raw, stack_dataframe=True)
    len_data = len(pos_raw)
    num_bins = len_data // 22
    res_tmp = kwo.calc_ic_bin_info(pos_raw, fwd_raw, bins=num_bins)
    res_dt["ir"] = res_tmp["mean"] / (res_tmp["std"] + 1e-8) * np.sqrt(252 / 22)

    # cross_section
    res_tmp = kwo.calc_cross_section_ic_info(pos_raw, fwd_raw)
    res_dt["cs_ic"] = res_tmp["mean"]
    res_dt["cs_ir"] = res_tmp["mean"] / (res_tmp["std"] + 1e-8) * np.sqrt(252)

    # sharpe, tvr, exposure
    res_dt["sharpe"] = calc_sharpe(pos_raw, cost=cost)
    res_dt["tvr"] = kwo.calc_tvr(pos_one)
    res_dt["exposure"] = basic.calc_exposure(pos_one)
    res_dt["exposure_count"] = basic.calc_exposure_count(pos_one)

    # psi
    len_feat = len(pos_raw)
    is_feat = pos_raw.iloc[0 : len_feat // 2]
    os_feat = pos_raw.iloc[len_feat // 2 :]
    res_dt["psi"] = basic.calc_feat_psi(is_feat, os_feat)

    # win rate
    res_dt["ic_win_rate"] = basic.calc_win_rate(pos_one, fwd_one, mode="ic")
    res_dt["ret_win_rate"] = basic.calc_win_rate(pos_one, fwd_one, mode="ret")

    # bootstrap ic
    len_boot = min(100, len(pos_raw) // 2)
    dt_ic_boot = bootstrap.get_boot_stat(
        pos_raw,
        fwd_raw,
        num_boot=100,
        len_boot=len_boot,
        mode="ic",
        boot_method="interval",
    )
    res_dt["ic_ib_mean"] = dt_ic_boot["boot_mean"]
    res_dt["ic_ib_kurt"] = dt_ic_boot["boot_kurt"]
    res_dt["ic_ib_skew"] = dt_ic_boot["boot_skew"]
    res_dt["ic_ib_mos"] = (
        dt_ic_boot["boot_mean"]
        / (dt_ic_boot["boot_std"] + 1e-8)
        * np.sqrt(252 / len_boot)
    )

    # bootstrap sharpe
    len_boot = min(100, len(pos_raw) // 2)
    se_ret = kwo.calc_ret_from_feat(pos_one, fwd_one, fee=4e-4)
    dt_sharpe_boot = bootstrap.get_boot_stat(
        se_ret, num_boot=1000, len_boot=len_boot, mode="sharpe", boot_method="interval"
    )
    res_dt["sharpe_ib_mean"] = dt_sharpe_boot["boot_mean"]
    res_dt["sharpe_ib_kurt"] = dt_sharpe_boot["boot_kurt"]
    res_dt["sharpe_ib_skew"] = dt_sharpe_boot["boot_skew"]
    res_dt["sharpe_ib_mos"] = (
        dt_sharpe_boot["boot_mean"]
        / (dt_sharpe_boot["boot_std"] + 1e-8)
        * np.sqrt(252 / len_boot)
    )

    # white noise mos
    noise_ic_std = basic.get_noise_ic_std(pos_raw, num_sample=100)
    res_dt["noise_ic_mos"] = dt_ic_boot["boot_mean"] / (noise_ic_std + 1e-8)

    return res_dt


def calc_pnl(feat: pd.DataFrame, cost: float = 4e-4):
    pos = kwo.calc_pos(feat)
    pos = pos.rolling(config.fwd_day).mean()
    cur_fwd = env.fwd_one.reindex(pos.index)
    se_ret = (pos * cur_fwd).sum(axis=1)
    mv_pos = pos - pos.shift()
    mv_pos = mv_pos.abs()
    se_cost = cost * mv_pos.sum(axis=1)
    se_ret = se_ret - se_cost
    se_pnl = se_ret + 1
    se_pnl = se_pnl.cumprod()
    return se_pnl


def show_pnl(feat: pd.DataFrame, cost: float = 4e-4, save_path=None):
    se_pnl = calc_pnl(feat, cost)
    se_pnl = kwo.transform_daily_index_int_to_datetime(se_pnl)
    plt.figure(dpi=144)
    se_pnl.plot()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
