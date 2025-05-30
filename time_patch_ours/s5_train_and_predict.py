import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import feather
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

time_window_size_map = {15 * 60: '15m', 60 * 60: '1h', 6 * 3600: '6h'}


@dataclass
class Config:
    data_path: str = field(default=r"D:/competition_data/sample_data_1000/type_A", init=False)
    processed_time_point_data_path: str = field(default=r"D:/m2mfp_data/time_point_data/type_A", init=False)
    combined_sn_feature_data_path: str = field(default=r"D:/m2mfp_data/combined_sn_feature/type_A", init=False)
    ticket_path = r"D:/competition_data/competition_data_final/ticket.csv"
    train_data_path = f"D:/m2mfp_data/time_patch_ours/train_data/type_A"
    test_data_path = f"D:/m2mfp_data/time_patch_ours/test_data/type_A"
    result_path = f"D:/m2mfp_data/results/type_A"


train_date_range = ("2024-01-01", "2024-06-01")
test_date_range = ("2024-06-01", "2024-10-01")


def train_model(train_data_path):
    train_pos = feather.read_dataframe(f'{train_data_path}/positive_train.csv')
    train_neg = feather.read_dataframe(f'{train_data_path}/negative_train.csv')
    train_neg['label'] = 0
    train_all = pd.concat([train_pos, train_neg])
    train_all.fillna(0, inplace=True)
    train_all = train_all.sample(frac=1, random_state=2024)
    print("Shape of training set:", train_all.shape)

    set_v5 = set([f"sn_{i}" for i in range(1, 65670)])
    train_all = train_all[train_all.index.get_level_values(0).isin(set_v5)]

    use_features = train_all.columns
    use_features = [i for i in use_features if i != "LogTime" and i != "ReportTime"]
    use_features = [i for i in use_features if "ce_log_num" not in i]

    train_all = train_all[use_features]
    train_all = train_all.sort_index(axis=1)

    train_all["sn_name"] = train_all.index.get_level_values(0)
    train_all.drop(columns=["sn_name"], inplace=True)

    LGB_MODEL_PARAMS = {"learning_rate": 0.02, "n_estimators": 500, "max_depth": 8,
                        'num_leaves': 20, 'min_child_samples': 20, 'verbose': 1,
                        'importance_type': 'gain'}
    model = lgb.LGBMClassifier(**LGB_MODEL_PARAMS)
    model.fit(train_all.drop(columns=['label']), train_all['label'])

    # feature_importance = model.booster_.feature_importance(importance_type='gain')
    # importance_df = pd.DataFrame({
    #     'Feature': model.feature_name_,
    #     'Importance (Gain)': feature_importance
    # })
    # importance_df = importance_df.sort_values(by='Importance (Gain)', ascending=False)
    # importance_df.to_csv('feature_importance_gain.csv', index=False)

    return model


def predict(test_data_path, model):
    result_all = {}
    for test_file in os.listdir(test_data_path):
        test_df = feather.read_dataframe(f'{test_data_path}/{test_file}')
        test_df["sn_name"] = test_df.index.get_level_values(0)
        test_df["log_time"] = test_df.index.get_level_values(1)

        test_df = test_df[model.feature_name_]

        predict_result = model.predict_proba(test_df)

        index_list = list(test_df.index)
        for i in tqdm(range(len(index_list)), test_file):
            p_s = predict_result[i][1]
            if p_s < 0.1:
                continue
            sn = index_list[i][0]
            sn_t = datetime.fromtimestamp(index_list[i][1])
            if sn not in result_all:
                result_all[sn] = [(sn_t, p_s)]
            else:
                result_all[sn].append((sn_t, p_s))
                # result_all[sn].append((sn_t, max(p_s, result_all[sn][-1][1])))
    return result_all


def train_and_predict():
    train_data_path = Config.train_data_path
    test_data_path = Config.test_data_path

    model = train_model(train_data_path)
    result_all = predict(test_data_path, model)

    with open(os.path.join(Config.result_path, "time_patch_ours.pkl"), 'wb') as f:
        pickle.dump(result_all, f)

os.makedirs(Config.result_path, exist_ok=True)
ticket = pd.read_csv(Config.ticket_path)
ticket = ticket[ticket['sn_type'] == "A"]

if __name__ == "__main__":
    train_and_predict()
