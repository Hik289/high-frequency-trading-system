import numpy as np
import pandas as pd
import sys
import os
import warnings
import itertools
from importlib import reload

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("../")
from utils import kiwi_operators as kwo
from utils import config

reload(kwo)


def judge_info(info_dt):
    if np.isnan(list(info_dt.values())).any():
        return False

    if info_dt["ic"] < config.search_limits["ic"]:
        return False
    if info_dt["ir"] < config.search_limits["ir"]:
        return False

    if info_dt["cs_ic"] < config.search_limits["ic"]:
        return False
    if info_dt["cs_ir"] < config.search_limits["ir"]:
        return False

    if info_dt["sharpe"] < config.search_limits["sharpe"]:
        return False

    if info_dt["tvr"] < config.search_limits["tvr"]:
        return False
    if info_dt["psi"] > config.search_limits["psi"]:
        return False

    return True


def compare_info(info_1, info_2):
    if np.isnan(list(info_1.values())).any():
        return False
    if np.isnan(list(info_2.values())).any():
        return False

    if info_2["ic"] < config.compare_ratio * info_1["ic"]:
        return False
    if info_2["ir"] < config.compare_ratio * info_1["ir"]:
        return False

    if info_2["cs_ic"] < config.compare_ratio * info_1["cs_ic"]:
        return False
    if info_2["cs_ir"] < config.compare_ratio * info_1["cs_ir"]:
        return False

    if info_2["sharpe"] < config.compare_ratio * info_1["sharpe"]:
        return False

    return True


def get_adjoint_params(param):
    res_lst = list()
    for idx in range(len(param)):
        old_wid = list()

        wid = max(1, int(config.adjoint_ratio * param[idx]))
        if wid not in old_wid:
            old_wid.append(wid)
            wid = np.abs(wid)
            res_tmp = param[:]
            res_tmp[idx] = max(param[idx] - wid, 1)
            res_lst.append(res_tmp)
            res_tmp = param[:]
            res_tmp[idx] = param[idx] + wid
            res_lst.append(res_tmp)

        wid = max(1, int(config.adjoint_ratio * param[idx] / 3))
        if wid not in old_wid:
            old_wid.append(wid)
            wid = np.abs(wid)
            res_tmp = param[:]
            res_tmp[idx] = max(param[idx] - wid, 1)
            res_lst.append(res_tmp)
            res_tmp = param[:]
            res_tmp[idx] = param[idx] + wid
            res_lst.append(res_tmp)

        wid = max(1, int(config.adjoint_ratio * param[idx] * 2 / 3))
        if wid not in old_wid:
            old_wid.append(wid)
            wid = np.abs(wid)
            res_tmp = param[:]
            res_tmp[idx] = max(param[idx] - wid, 1)
            res_lst.append(res_tmp)
            res_tmp = param[:]
            res_tmp[idx] = param[idx] + wid
            res_lst.append(res_tmp)

    return res_lst
