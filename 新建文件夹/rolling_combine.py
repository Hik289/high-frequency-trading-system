import pandas as pd
import numpy as np
import joblib

# kwo: customized functions
# config: project configuration
# env: raw data storage
from utils import kiwi_operators as kwo
from utils import config
from utils import data_environment as env
from utils.combiner import Combiner
from utils import backtester
from importlib import reload
from glob import glob
import os
import shutil

os.environ["OMP_NUM_THREADS"] = "1"

reload(kwo)
reload(config)

print("environment loaded")

pos_lst = list()
for train_date_pair, pred_date_pair in zip(
    config.train_date_pairs, config.pred_date_pairs
):
    train_date_beg = train_date_pair[0]
    train_date_end = train_date_pair[1]
    pred_date_beg = pred_date_pair[0]
    pred_date_end = pred_date_pair[1]

    print("----------------------------------------------------")
    print(f"train: from {train_date_beg} to {train_date_end}")
    print(f"predict from {pred_date_beg} to {pred_date_end}")

    # copy feature functions
    shutil.rmtree(f"{config.dir_selected_feat}", ignore_errors=True)
    os.makedirs(f"{config.dir_selected_feat}", exist_ok=False)
    dir_from = (
        f"{config.dir_rolling_selected_feat}/{train_date_beg}_{train_date_end}/func"
    )
    shutil.copytree(dir_from, f"{config.dir_selected_feat}/func")

    # regenerate features
    os.makedirs(f"{config.dir_selected_feat}/data", exist_ok=False)
    path_lst = glob(f"{config.dir_selected_feat}/func/*.pkl")
    if len(path_lst) < 1:
        continue
    name_lst = kwo.get_names_from_paths(path_lst)
    for name, path in zip(name_lst, path_lst):
        cur_func = joblib.load(path)
        cur_feat = cur_func(env.data)
        cur_feat.to_pickle(f"{config.dir_selected_feat}/data/{name}.pkl")

    # combine features
    combiner = Combiner()
    combiner.train(train_date_beg, train_date_end, verbose=True)
    cur_pos = combiner.predict(pred_date_beg, pred_date_end)
    cur_sharpe = backtester.calc_sharpe(cur_pos, cost=config.cost)
    pos_lst.append(cur_pos)
    print("period task finished with sharpe %.3f" % cur_sharpe)

full_pos = pd.concat(pos_lst, axis=0)
backtester.show_pnl(
    full_pos, cost=config.cost, save_path=f"{config.dir_rolling_res}/pnl.png"
)

dt_metric = backtester.get_metrics(full_pos)
cur_sharpe = backtester.calc_sharpe(full_pos, cost=config.cost)
res_metric = list()
for k in dt_metric.keys():
    res_metric.append([k, dt_metric[k]])
res_metric = pd.DataFrame(res_metric, columns=["metric", "value"])
res_metric.to_csv(f"{config.dir_rolling_res}/metric.csv")

full_pos.to_pickle(f"{config.dir_rolling_res}/pos.pkl")
full_pos = kwo.transform_daily_index_int_to_str(full_pos)
full_pos.to_pickle(f"{config.dir_rolling_res}/pos.csv")

print("combination finished")
print("sharpe: %.3f" % cur_sharpe)
print(f"PnL figure saved as {config.dir_rolling_res}/pnl.png")
print(f'metrics saved as {config.dir_rolling_res}/metric.csv"')
print(f"position saved as {config.dir_rolling_res}/pos.csv")
print(f"position (integer indices) saved as {config.dir_rolling_res}/pos.pkl")
