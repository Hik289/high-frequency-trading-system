import shutil
import pandas as pd
import numpy as np
import sys
import os
from . import kiwi_operators as kwo
from importlib import reload
from glob import glob
from .feature_selection import packed_selectors as ps
from . import config
from multiprocessing import Pool
from . import data_environment as env
from .feature_selection.mutual_correlation import calc_correlation_matrix
from .feature_selection.mutual_correlation import filter_mutual_correlation
from . import split_data as sd
from .feature_selection.gbm_classification import get_gbm_classification_info

# reload(kwo)
# reload(config)
# reload(sd)
# reload(env)

os.environ["OMP_NUM_THREADS"] = "1"


def part_stats(param):
    return ps.get_full_info(feat=param[0], fwd_ret=param[1], fwd_day=param[2])


def calc_stats(
    feat_lst,
    fwd_ret: pd.DataFrame,
    fwd_day: int = 1,
    out_path: str = None,
    name_lst=None,
):
    param_lst = [[elem, fwd_ret, fwd_day] for elem in feat_lst]
    pool = Pool(os.cpu_count() * 2 // 3)
    info_lst = pool.map(part_stats, param_lst)
    pool.close()
    pool.join()
    del pool
    info_items = sorted(list(info_lst[0].keys()))
    res_lst = list()
    for cur_info in info_lst:
        cur_res = [cur_info[info_item] for info_item in info_items]
        res_lst.append(cur_res)
    df_res = pd.DataFrame(res_lst)
    df_res.columns = info_items

    if name_lst is not None:
        df_res.index = name_lst
    if out_path is not None:
        df_res.to_pickle(out_path)
    return df_res


class FeatureSelector(object):
    def __init__(self, train_date_beg: int, train_date_end: int):
        self.train_date_beg = train_date_beg
        self.train_date_end = train_date_end
        self.train_fwd_ret = kwo.get_partial_dataframe_by_date(
            env.fwd_ret, train_date_beg, train_date_end
        )

    def get_train_dataframe(self, dataframe: pd.DataFrame):
        return kwo.get_partial_dataframe_by_date(
            dataframe, date_beg=self.train_date_beg, date_end=self.train_date_end
        )

    def calc_train_stats(self, name_lst):
        paths = [f"{config.dir_feat}/data/{elem}.pkl" for elem in name_lst]
        feats = [pd.read_pickle(elem) for elem in paths]
        train_feats = [self.get_train_dataframe(elem) for elem in feats]
        out_path = f"{config.dir_feat}/stats.pkl"
        df_res = calc_stats(
            feat_lst=train_feats,
            fwd_ret=self.train_fwd_ret,
            fwd_day=config.fwd_day,
            out_path=out_path,
            name_lst=name_lst,
        )
        return df_res

    def filter_by_stats(self, name_lst, df_stats=None):
        if df_stats is None:
            df_stats = self.calc_train_stats(name_lst)
        else:
            df_stats = df_stats.loc[name_lst]

        df_stats = df_stats[df_stats["ic_ib_mean"] > config.select_limits["ic_ib_mean"]]
        df_stats = df_stats[df_stats["cs_ic"] > config.select_limits["cs_ic"]]
        df_stats = df_stats[df_stats["ic_ib_mos"] > config.select_limits["ic_ib_mos"]]
        df_stats = df_stats[df_stats["cs_ir"] > config.select_limits["cs_ir"]]
        df_stats = df_stats[df_stats["psi"] < config.select_limits["psi"]]
        df_stats = df_stats[
            df_stats["exposure"].abs() < config.select_limits["exposure"]
        ]
        df_stats = df_stats[
            df_stats["exposure_count"].abs() < config.select_limits["exposure_count"]
        ]
        df_stats = df_stats[
            df_stats["sharpe_ib_mean"] > config.select_limits["sharpe_ib_mean"]
        ]
        df_stats = df_stats[
            df_stats["sharpe_ib_mos"] > config.select_limits["sharpe_ib_mos"]
        ]

        res_names = list(df_stats.index)
        return sorted(res_names)

    def filter_by_mul_corr(self, name_lst, eval_lst, max_corr):
        paths = [f"{config.dir_feat}/data/{elem}.pkl" for elem in name_lst]
        feats = [pd.read_pickle(elem) for elem in paths]
        train_feats = [self.get_train_dataframe(elem) for elem in feats]
        corr_mat = calc_correlation_matrix(train_feats)
        res_idx_lst = filter_mutual_correlation(
            correlation_matrix=corr_mat, eval_by=eval_lst, max_corr=max_corr
        )
        res_names = [name_lst[elem] for elem in res_idx_lst]
        return res_names

    def get_stacked_train_feat(self, name_lst):
        paths = [f"{config.dir_feat}/data/{elem}.pkl" for elem in name_lst]
        feats = [pd.read_pickle(elem) for elem in paths]
        train_feats = [self.get_train_dataframe(elem) for elem in feats]
        df_res = kwo.stack_feats(feats=train_feats, feat_names=name_lst)
        df_res.to_pickle(f"{config.dir_cache}/select_stacked.pkl")
        return df_res

    def filter_feats_by_auc(self, name_lst, max_auc: float = 0.7, verbose: bool = True):
        stacked_feat = self.get_stacked_train_feat(name_lst=name_lst)
        col_lst = list(stacked_feat.columns)
        col_lst.remove("investment")
        stacked_feat = stacked_feat[col_lst]
        data_lst = sd.split_dataframe_plain(stacked_feat, by="time", ratio_lst=[1, 1])
        data_0 = data_lst[0]
        data_1 = data_lst[1]
        col_lst.remove("time")
        data_0 = np.array(data_0[col_lst])
        data_1 = np.array(data_1[col_lst])

        remained = list(range(data_0.shape[1]))
        cur_auc = 1
        step_cnt = 0
        while len(remained) > 0 and cur_auc > max_auc:
            step_cnt += 1
            cur_data_0 = data_0[:, remained]
            cur_data_1 = data_1[:, remained]
            cur_res = get_gbm_classification_info(data_0=cur_data_0, data_1=cur_data_1)
            cur_auc = cur_res["auc"]
            if cur_auc > max_auc:
                removed_idx = remained[cur_res["idx_max"]]
                removed_importance = np.max(cur_res["importance"]) / np.sum(
                    cur_res["importance"]
                )
                remained.pop(cur_res["idx_max"])
                if verbose:
                    print(
                        "step--%d auc--%.3f removed-idx--%d removed-importance--%.3f remained-num--%d"
                        % (
                            step_cnt,
                            cur_auc,
                            removed_idx,
                            removed_importance,
                            len(remained),
                        )
                    )
            else:
                if verbose:
                    print("final auc: %.3f" % cur_auc)

        res_lst = [name_lst[elem] for elem in remained]
        return res_lst

    def filter_features(self, name_lst=None, verbose: bool = False):
        if name_lst is None:
            path_lst = glob(f"{config.dir_feat}/data/*.pkl")
            name_lst = kwo.get_names_from_paths(path_lst)

        if verbose:
            print("Initial number of features: %d." % len(name_lst))
            # print("---------------------------------------------")
            # print("Criteria: ")
            # for k, v in zip(config.select_limits.keys(), config.select_limits.values()):
            #     print(k, ":", v)
            # print("---------------------------------------------")
            # print()
            # print("Filtering by feature statistics...")
        df_stats = self.calc_train_stats(name_lst)
        name_lst = self.filter_by_stats(name_lst=name_lst, df_stats=df_stats)
        if verbose:
            print("Filtered by feature statistics, %d remained." % len(name_lst))

        # if verbose:
        #     print()
        #     print("Filtering by mutual correlation...")
        df_stats = df_stats.loc[name_lst]
        eval_lst = df_stats["cs_ir"]
        name_lst = self.filter_by_mul_corr(
            name_lst=name_lst,
            eval_lst=eval_lst,
            max_corr=config.select_limits["mul_corr"],
        )
        if verbose:
            print("Filtered by mutual correlation, %d remained." % len(name_lst))

        # if verbose:
        #     print()
        #     print("Filtering by GBM classification auc...")
        # name_lst = self.filter_feats_by_auc(
        #     name_lst=name_lst,
        #     max_auc=config.select_limits["classify_auc"],
        #     verbose=verbose,
        # )
        # if verbose:
        #     print("Filtered by GBM classification auc, %d remained." % len(name_lst))

        return name_lst

    def filter_copy_features(self, name_lst=None, verbose: bool = False):
        name_lst = self.filter_features(name_lst=name_lst, verbose=verbose)
        os.makedirs(f"{config.dir_selected_feat}/data", exist_ok=True)
        os.makedirs(f"{config.dir_selected_feat}/func", exist_ok=True)

        for elem in name_lst:
            old_path = f"{config.dir_feat}/data/{elem}.pkl"
            new_path = f"{config.dir_selected_feat}/data/{elem}.pkl"
            shutil.copyfile(old_path, new_path)

            old_path = f"{config.dir_feat}/func/{elem}.pkl"
            new_path = f"{config.dir_selected_feat}/func/{elem}.pkl"
            shutil.copyfile(old_path, new_path)

        return name_lst

