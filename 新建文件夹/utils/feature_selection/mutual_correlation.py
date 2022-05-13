#%%
import pandas as pd
import numpy as np
import sys
import os
from .. import kiwi_operators as kwo
from importlib import reload
from multiprocessing import Pool

reload(kwo)

os.environ["OMP_NUM_THREADS"] = "1"


def part_series_ic(param):
    return kwo.calc_ic(param[0], param[1])


def part_dataframe_ic(param):
    return kwo.calc_ic(param[0], param[1], stack_dataframe=True)


def calc_correlation_matrix(feats):
    corr_mat = np.full((len(feats), len(feats)), np.nan)
    input_indices = list()
    input_feats = list()
    for idx in range(len(feats)):
        for jdx in range(len(feats)):
            if idx < jdx:
                input_indices.append([idx, jdx])
                input_feats.append([feats[idx], feats[jdx]])
    pool = Pool(os.cpu_count() * 2 // 3)
    if type(feats[0]) == pd.Series:
        outputs = pool.map(part_series_ic, input_feats)
    elif type(feats[0]) == pd.DataFrame:
        outputs = pool.map(part_dataframe_ic, input_feats)
    else:
        raise ValueError("Invalid inputs.")
    pool.close()
    pool.join()
    del pool

    for input, output in zip(input_indices, outputs):
        idx = input[0]
        jdx = input[1]
        corr_mat[idx, jdx] = output
        corr_mat[jdx, idx] = output
    for idx in range(len(feats)):
        corr_mat[idx, idx] = 1
    return corr_mat


def filter_mutual_correlation(correlation_matrix, eval_by, max_corr: float = 0.5):
    corr_mat = correlation_matrix.copy()
    corr_mat = np.abs(corr_mat)
    eval_lst = eval_by.copy()

    rm_lst = list()
    for idx in range(len(corr_mat)):
        corr_mat[idx, idx] = 0
    for idx in range(len(corr_mat)):
        for jdx in range(len(corr_mat)):
            if corr_mat[idx, jdx] > max_corr:
                if eval_lst[idx] > eval_lst[jdx]:
                    corr_mat[:, jdx] = 0
                    corr_mat[jdx, :] = 0
                    rm_lst.append(jdx)
                else:
                    corr_mat[:, idx] = 0
                    corr_mat[idx, :] = 0
                    rm_lst.append(idx)
                    break

    rm_set = set(rm_lst)
    all_set = set(range(len(corr_mat)))
    res_set = all_set - rm_set
    res_lst = sorted(list(res_set))
    return res_lst
