#%%
import pandas as pd
import numpy as np
import sys
import toad
import os

from . import basic
from . import bootstrap as bs

sys.path.append("..")
from .. import kiwi_operators as kwo
from importlib import reload

# reload(kwo)
# reload(bs)

os.environ["OMP_NUM_THREADS"] = "1"


def get_basic_info(feat: pd.DataFrame, fwd_ret: pd.DataFrame, fwd_day: int = 1):
    idx_1 = set(feat.index)
    idx_2 = set(fwd_ret.index)
    co_idx = idx_1.intersection(idx_2)
    co_idx = sorted(list(co_idx))
    feat = feat.reindex(co_idx)
    fwd_ret = fwd_ret.reindex(co_idx)
    res_dt = dict()

    # time_series
    res_dt["ic"] = kwo.calc_ic(feat, fwd_ret, stack_dataframe=True)
    len_data = len(feat)
    num_bins = len_data // 22
    res_tmp = kwo.calc_ic_bin_info(feat, fwd_ret, bins=num_bins)
    res_dt["ir"] = res_tmp["mean"] / (res_tmp["std"] + 1e-8) * np.sqrt(252 / 22)

    # cross_section
    res_tmp = kwo.calc_cross_section_ic_info(feat, fwd_ret)
    res_dt["cs_ic"] = res_tmp["mean"]
    res_dt["cs_ir"] = res_tmp["mean"] / (res_tmp["std"] + 1e-8) * np.sqrt(252)

    # sharpe
    res_dt["sharpe"] = kwo.calc_sharpe_from_feat(feat, fwd_ret) / np.sqrt(fwd_day)

    res_dt["tvr"] = kwo.calc_tvr(feat)
    res_dt["pos_concentration"] = basic.calc_pos_concentration(feat)
    res_dt["ret_concentration"] = basic.calc_ret_concentration(feat, fwd_ret)
    res_dt["exposure"] = basic.calc_exposure(feat)
    res_dt["exposure_count"] = basic.calc_exposure_count(feat)

    len_feat = len(feat)
    is_feat = feat.iloc[0 : len_feat // 2]
    os_feat = feat.iloc[len_feat // 2 :]
    res_dt["psi"] = basic.calc_feat_psi(is_feat, os_feat)
    res_dt["ic_win_rate"] = basic.calc_win_rate(feat, fwd_ret, mode="ic")
    res_dt["ret_win_rate"] = basic.calc_win_rate(feat, fwd_ret, mode="ret")

    return res_dt


def get_multifold_min_info(
    feat: pd.DataFrame, fwd_ret: pd.DataFrame, num_folds: int = 5, fwd_day: int = 1
):
    feat = feat.dropna(how="all")
    fwd_ret = fwd_ret.dropna(how="all")
    idx_1 = set(feat.index)
    idx_2 = set(fwd_ret.index)
    co_idx = idx_1.intersection(idx_2)
    co_idx = sorted(list(co_idx))
    feat = feat.reindex(co_idx)
    fwd_ret = fwd_ret.reindex(co_idx)
    res_dt = dict()
    len_data = len(co_idx)
    wid_data = len_data // num_folds

    sharpe = kwo.calc_sharpe_from_feat(feat, fwd_ret) / np.sqrt(fwd_day)

    feat_lst = [
        feat.iloc[idx * wid_data : (idx + 1) * wid_data :] for idx in range(num_folds)
    ]
    fwd_lst = [
        fwd_ret.iloc[idx * wid_data : (idx + 1) * wid_data :]
        for idx in range(num_folds)
    ]

    ic_lst = [
        kwo.calc_ic(feat_lst[idx], fwd_lst[idx], stack_dataframe=True)
        for idx in range(num_folds)
    ]
    cs_ic_lst = list()
    cs_ir_lst = list()
    for idx in range(num_folds):
        res_tmp = kwo.calc_cross_section_ic_info(feat_lst[idx], fwd_lst[idx])
        cs_ic_lst.append(res_tmp["mean"])
        cs_ir_lst.append(res_tmp["mean"] / (res_tmp["std"] + 1e-8) * np.sqrt(252))

    sharpe_lst = [
        kwo.calc_sharpe_from_feat(feat_lst[idx], fwd_lst[idx])
        for idx in range(num_folds)
    ]

    if sharpe < 0:
        ic_lst = [-elem for elem in ic_lst]
        cs_ic_lst = [-elem for elem in cs_ic_lst]
        cs_ir_lst = [-elem for elem in cs_ir_lst]
        sharpe_lst = [-elem for elem in sharpe_lst]

    ic_win_rate_lst = [
        basic.calc_cs_win_rate(feat_lst[idx], fwd_lst[idx], mode="ic")
        for idx in range(num_folds)
    ]
    ret_win_rate_lst = [
        basic.calc_cs_win_rate(feat_lst[idx], fwd_lst[idx], mode="ret")
        for idx in range(num_folds)
    ]

    res_dt = dict()
    res_dt["mfm_ic"] = np.min(ic_lst)
    res_dt["mfm_cs_ic"] = np.min(cs_ic_lst)
    res_dt["mfm_cs_ir"] = np.min(cs_ir_lst)
    res_dt["mfm_sharpe"] = np.min(sharpe_lst)
    res_dt["mfm_ic_win_rate"] = np.min(ic_win_rate_lst)
    res_dt["mfm_ret_win_rate"] = np.min(ret_win_rate_lst)

    return res_dt


def get_resample_info(feat: pd.DataFrame, fwd_ret: pd.DataFrame, fwd_day: int = 1):
    idx_1 = set(feat.index)
    idx_2 = set(fwd_ret.index)
    co_idx = idx_1.intersection(idx_2)
    co_idx = sorted(list(co_idx))
    feat = feat.reindex(co_idx)
    fwd_ret = fwd_ret.reindex(co_idx)
    res_dt = dict()

    len_boot = min(100, len(feat) // 2)

    dt_ic_boot = bs.get_boot_stat(
        feat,
        fwd_ret,
        num_boot=100,
        len_boot=len_boot,
        mode="ic",
        boot_method="interval",
    )
    res_dt["ic_ib_mean"] = dt_ic_boot["boot_mean"]
    res_dt["ic_ib_kurt"] = dt_ic_boot["boot_kurt"]
    res_dt["ic_ib_skew"] = dt_ic_boot["boot_skew"]
    res_dt["ic_ib_mos"] = (
        dt_ic_boot["boot_mean"] / (dt_ic_boot["boot_std"] + 1e-8) * np.sqrt(252 / 100)
    )

    se_ret = kwo.calc_ret_from_feat(feat, fwd_ret)
    dt_sharpe_boot = bs.get_boot_stat(
        se_ret, num_boot=1000, len_boot=len_boot, mode="sharpe", boot_method="interval"
    )
    res_dt["sharpe_ib_mean"] = dt_sharpe_boot["boot_mean"] / np.sqrt(fwd_day)
    res_dt["sharpe_ib_kurt"] = dt_sharpe_boot["boot_kurt"]
    res_dt["sharpe_ib_skew"] = dt_sharpe_boot["boot_skew"]
    res_dt["sharpe_ib_mos"] = (
        dt_sharpe_boot["boot_mean"]
        / (dt_sharpe_boot["boot_std"] + 1e-8)
        * np.sqrt(252 / 100)
    )

    noise_ic_std = basic.get_noise_ic_std(feat, num_sample=100)
    res_dt["noise_ic_mos"] = dt_ic_boot["boot_mean"] / (noise_ic_std + 1e-8)

    return res_dt


def get_full_info(feat: pd.DataFrame, fwd_ret: pd.DataFrame, fwd_day: int = 1):
    basic_dt = get_basic_info(feat, fwd_ret, fwd_day)
    resample_dt = get_resample_info(feat, fwd_ret, fwd_day)
    basic_dt.update(resample_dt)
    return basic_dt
