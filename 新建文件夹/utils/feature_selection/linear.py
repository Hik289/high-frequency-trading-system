#%%
import pandas as pd
import numpy as np
import sys
import toad
import os
from scipy.stats import pearsonr

sys.path.append("..")
from .. import kiwi_operators as kwo
from importlib import reload

reload(kwo)

os.environ["OMP_NUM_THREADS"] = "1"


def get_corr_info(
    data_1, data_2,
):
    data_1 = data_1.replace([np.inf, -np.inf], np.nan)
    data_2 = data_2.replace([np.inf, -np.inf], np.nan)
    indices_1 = set(data_1.index)
    indices_2 = set(data_2.index)
    co_indices = sorted(list(indices_1.intersection(indices_2)))
    data_1 = data_1.reindex(co_indices)
    data_2 = data_2.reindex(co_indices)

    # ======================================================
    # non-pos
    # ======================================================
    # calculate stack info
    se_1 = data_1.stack()
    se_2 = data_2.stack()
    df_both = pd.concat([se_1, se_2], axis=1).dropna(how="any")
    arr_1 = np.array(df_both.iloc[:, 0])
    arr_2 = np.array(df_both.iloc[:, 1])
    res_stack = pearsonr(arr_1, arr_2)
    print(res_stack)

