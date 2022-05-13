import numpy as np
import pandas as pd
import joblib

from .. import kiwi_operators as kwo
from .. import config
from .. import data_environment as env
from importlib import reload
from glob import glob
import os
import shutil
from sklearn.cluster import KMeans

reload(kwo)
reload(config)
reload(env)


def get_row_cluster_mean(val_lst, label_lst):
    dt_sum = dict()
    dt_cnt = dict()
    len_data = len(val_lst)
    for idx in range(len_data):
        cur_val = val_lst[idx]
        cur_label = label_lst[idx]
        if np.isnan(cur_val):
            continue
        if cur_label in dt_sum.keys():
            dt_sum[cur_label] += cur_val
            dt_cnt[cur_label] += 1
        else:
            dt_sum[cur_label] = cur_val
            dt_cnt[cur_label] = 1
    res_lst = [dt_sum[elem] / dt_cnt[elem] for elem in label_lst]
    return np.array(res_lst)


class ClusterMeanModifier(object):
    def __init__(
        self, num_train_rows: int = 22, n_clusters: int = 6, mean_ratio: float = 0.3
    ):
        self.num_train_rows = num_train_rows
        self.n_clusters = n_clusters
        self.mean_ratio = mean_ratio
        self.kmeans = KMeans(self.n_clusters)

    def get_row_cluster_label(self, dataframe: pd.DataFrame, row_idx: int):
        if row_idx < self.num_train_rows:
            return np.full(dataframe.shape[1], 0)

        data = np.array(
            dataframe.iloc[row_idx - self.num_train_rows : row_idx]
        ).transpose()
        self.kmeans.fit(data)
        return self.kmeans.labels_

    def transform(self, dataframe: pd.DataFrame):
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        dataframe = dataframe.fillna(0)
        cluster_lst = list()
        for row_idx in range(len(dataframe)):
            cluster_lst.append(self.get_row_cluster_label(dataframe, row_idx))

        arr_data = np.array(dataframe)
        arr_cluster = np.array(cluster_lst)

        res_lst = list()
        for idx in range(len(arr_data)):
            cur_res = get_row_cluster_mean(arr_data[idx], arr_cluster[idx])
            res_lst.append(cur_res)
        df_res = pd.DataFrame(res_lst, index=dataframe.index, columns=dataframe.columns)
        df_res = self.mean_ratio * df_res + (1 - self.mean_ratio) * dataframe
        return df_res

