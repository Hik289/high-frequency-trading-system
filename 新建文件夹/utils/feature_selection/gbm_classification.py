import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"


def get_gbm_classification_info(data_0: np.array, data_1: np.array):
    x_data = list()
    y_data = list()
    for elem in data_0:
        x_data.append(elem)
        y_data.append(0)
    for elem in data_1:
        x_data.append(elem)
        y_data.append(1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, shuffle=True
    )
    model = XGBClassifier(
        max_depth=7,
        gamma=0.01,
        learning_rate=0.05,
        reg_alpha=0.01,
        n_jobs=os.cpu_count() * 3 // 4,
    )

    # model = LGBMClassifier(
    #     max_depth=7,
    #     num_leaves=31,
    #     learning_rate=0.05,
    #     colsample_bytree=0.65,
    #     subsample=0.85,
    #     subsample_freq=5,
    #     reg_alpha=0.01,
    #     n_jobs=os.cpu_count() * 3 // 4,
    # )

    eval_set = [(x_test, y_test)]
    model.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        eval_metric="auc",
        verbose=False,
        early_stopping_rounds=20,
    )

    importances = model.feature_importances_
    p_test = model.predict(x_test)
    res_auc = roc_auc_score(y_test, p_test)
    sorted_id = sorted(
        range(len(importances)), key=lambda k: importances[k], reverse=True
    )
    res_dt = dict()
    res_dt["auc"] = res_auc
    res_dt["order"] = sorted_id
    res_dt["importance"] = importances
    res_dt["idx_max"] = sorted_id[0]
    del model
    return res_dt
