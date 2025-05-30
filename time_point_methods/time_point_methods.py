import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 获取项目根目录的绝对路径
project_root = Path(__file__).resolve().parents[1]  # 根据层级调整parents的值
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from utils.helpers import read_dataframe


def get_use_parity_feature_names():
    parity_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["F_max_pooling", "max_pooling_F"]:
            for F_function in ["bit_count", "bit_min_interval", "bit_max_interval",
                               "bit_max_consecutive_length", "bit_consecutive_length"]:
                parity_feature_names.append(f"parity_{row_column}wise_{G_function}_{F_function}")
    return parity_feature_names


features = get_use_parity_feature_names() + ['in_pos_parity_set',
                                             'retry_log_is_uncorrectable_error']


def all_digits_even(n):
    while n > 0:
        if (n % 16) % 2 != 0:
            return False
        n //= 16
    return True


def process_hex(num):
    # if num == 0:
    #     return "0"
    # while all_digits_even(num):
    #     num //= 2

    return num


# 自定义 F1 分数决策树
class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, *args, **kwargs):
        first_layer_feature = kwargs.pop('first_layer_feature', None)
        second_layer_feature = kwargs.pop('second_layer_feature', None)
        super().__init__(*args, **kwargs)
        self.first_layer_feature = first_layer_feature
        self.second_layer_feature = second_layer_feature

    def _gini(self, y):
        """计算 Gini 指数"""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _weighted_gini(self, y_left, y_right):
        """计算加权 Gini 指数"""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        return (n_left / n_total) * self._gini(y_left) + (n_right / n_total) * self._gini(y_right)

    def _split_criterion(self, X, y, depth):
        """找到最佳分割特征和阈值"""
        if depth == 0 and self.first_layer_feature is not None:
            feature = features.index(self.first_layer_feature)
            thresholds = np.unique(X[:, feature])
        elif depth == 1 and self.second_layer_feature is not None:
            feature = features.index(self.second_layer_feature)
            thresholds = np.unique(X[:, feature])
        else:
            best_gini = float('inf')
            best_split = (None, None)
            n_samples, n_features = X.shape
            n_features -= 1

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_mask = X[:, feature] == threshold
                    left_sns_set = set(X[left_mask][:, -1])
                    right_sns_set = all_sns_set - left_sns_set
                    left_mask = np.array([x in left_sns_set for x in X[:, -1]])
                    right_mask = ~left_mask

                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue

                    y_left = [1 if i in pos_sn_set else 0 for i in left_sns_set]
                    y_right = [1 if i in pos_sn_set else 0 for i in right_sns_set]

                    gini = self._weighted_gini(y_left, y_right)

                    if gini < best_gini:
                        best_gini = gini
                        best_split = (feature, threshold)

            return best_split

        best_gini = float('inf')
        best_split = (None, None)

        for threshold in thresholds:
            left_mask = X[:, feature] == threshold
            left_sns_set = set(X[left_mask][:, -1])
            right_sns_set = all_sns_set - left_sns_set
            left_mask = np.array([x in left_sns_set for x in X[:, -1]])
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = [1 if i in pos_sn_set else 0 for i in left_sns_set]
            y_right = [1 if i in pos_sn_set else 0 for i in right_sns_set]

            gini = self._weighted_gini(y_left, y_right)

            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold)

        return best_split

    def fit(self, X, y):
        """训练模型"""
        self.tree_ = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件 1: 如果所有样本属于同一类别
        if len(np.unique(y)) == 1:
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 停止条件 2: 如果达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            if len(y) == 0:
                return {'label': 0, 'class_counts': {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 找到最佳分割
        feature, threshold = self._split_criterion(X, y, depth)

        # 停止条件 3: 如果无法找到有效分割
        if feature is None:
            if len(y) == 0:
                return {'label': 0, 'class_counts': {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set), 1: len(set(X[:, -1]) & pos_sn_set)}
            return {'label': label, 'class_counts': class_counts}

        # 递归构建左右子树
        left_mask = X[:, feature] == threshold
        left_sns_set = set(X[left_mask][:, -1])
        right_sns_set = all_sns_set - left_sns_set
        left_mask = np.array([x in left_sns_set for x in X[:, -1]])
        right_mask = ~left_mask
        left_mask = X[:, feature] == threshold

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def predict(self, X):
        """预测"""
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree):
        """预测单个样本"""
        if 'label' in tree:  # 如果是叶子节点
            return tree['label']
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] == threshold:  # 如果样本的特征值等于阈值
            return self._predict_one(x, tree['left'])
        else:  # 如果样本的特征值大于阈值
            return self._predict_one(x, tree['right'])

    def predict_proba(self, X):
        """返回每个样本属于各个类别的概率"""
        return np.array([self._predict_proba_one(x, self.tree_) for x in X])

    def _predict_proba_one(self, x, tree):
        """返回单个样本的概率分布"""
        if 'class_counts' in tree:  # 如果是叶子节点
            class_counts = tree['class_counts']
            total_samples = sum(class_counts.values())  # 计算总样本数
            proba = [class_counts[0] / total_samples, class_counts[1] / total_samples]  # 计算概率
            return proba
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] == threshold:
            return self._predict_proba_one(x, tree['left'])
        else:
            return self._predict_proba_one(x, tree['right'])

    def get_tree(self):
        def replace_feature_names(node):
            if 'label' in node:
                return node
            feature_name = features[node['feature']]
            left_tree = replace_feature_names(node['left'])
            right_tree = replace_feature_names(node['right'])
            return {'feature': feature_name, 'threshold': node['threshold'], 'left': left_tree, 'right': right_tree}

        return replace_feature_names(self.tree_)

    def extract_rules(self, node=None, current_rule=None):
        if node is None:
            node = self.tree_
        if current_rule is None:
            current_rule = []

        if 'label' in node:
            if node['label'] == 1:
                return [current_rule]
            else:
                return []

        rules = []
        feature = features[node['feature']]
        threshold = node['threshold']

        left_rule = current_rule + [(feature, '==', threshold)]
        right_rule = current_rule + [(feature, '!=', threshold)]

        rules += self.extract_rules(node['left'], left_rule)
        rules += self.extract_rules(node['right'], right_rule)

        return rules


class RiskyCEClassifier:
    indices_left = np.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29])
    indices_right = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31])

    def predict(self, X):
        X_array = X.fillna(0).to_numpy(dtype=np.int64)
        bin_array = np.vectorize(lambda x: bin(x)[2:].zfill(32))(X_array)
        left_array = np.array([any(binary_str[i] == '1' for i in self.indices_left) for binary_str in bin_array])
        right_array = np.array([any(binary_str[i] == '1' for i in self.indices_right) for binary_str in bin_array])
        return left_array & right_array


class DQBeatPredictor:
    def predict(self, X):
        X_array = X.fillna(0).to_numpy(dtype=np.int64)
        bin_array = np.vectorize(lambda x: bin(x)[2:].zfill(32))(X_array)
        reshaped_array = np.array([list(s) for s in bin_array], dtype=int).reshape(-1, 8, 4)

        rows_with_ones = np.sum(np.any(reshaped_array == 1, axis=2), axis=1)
        cols_with_ones = np.sum(np.any(reshaped_array == 1, axis=1), axis=1)

        return (rows_with_ones > 1) & (cols_with_ones > 1)


def predict(test_data_path, model):
    result_all = {}
    for test_file in sorted(os.listdir(test_data_path),
                            key=lambda filename: int(filename.split('_')[-1].split('.')[0])):
        test_df = read_dataframe(f'{test_data_path}/{test_file}')
        test_df['in_pos_parity_set'] = test_df['RetryRdErrLogParity'].apply(
            lambda x: 1 if process_hex(int(x)) in anomaly_parities else 0)

        if model.__class__.__name__ == "RiskyCEClassifier":
            test_df = test_df["RetryRdErrLogParity"]
            predict_result = model.predict(test_df)
        elif model.__class__.__name__ == "DQBeatPredictor":
            test_df = test_df["RetryRdErrLogParity"]
            predict_result = model.predict(test_df)
        else:
            test_df = test_df[features]
            if method_name in ["time_point_gini", "time_point_lgb", "time_point_xgb"]:
                predict_result = model.predict(test_df.values)
            else:
                test_df["sn_name"] = test_df.index.get_level_values(0)
                X_test = test_df.values
                predict_result = model.predict(X_test)

        index_list = list(test_df.index)
        for i in tqdm(range(len(index_list)), test_file):
            p_s = predict_result[i]
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


if __name__ == "__main__":
    method_name_list = ["time_point_naive",
                        "time_point_parity_ours",
                        "time_point_four_bursts",
                        "time_point_risky_ce",
                        "time_point_dq_beat_predictor",
                        "time_point_gini",
                        "time_point_lgb",
                        "time_point_xgb"]

    method_name = "time_point_parity_ours"

    if method_name == "time_point_naive":
        with open("../resources/anomaly_parities.pkl", "rb") as f:
            anomaly_parities = set(pickle.load(f))
        first_layer_feature = 'in_pos_parity_set'
        second_layer_feature = None
        max_depth = 1
    elif method_name in ["time_point_parity_ours", "time_point_gini", "time_point_lgb", "time_point_xgb"]:
        anomaly_parities = set()
        first_layer_feature = None
        second_layer_feature = None
        max_depth = 4
    else:
        anomaly_parities = set()

    if method_name == "four_bursts":
        model = CustomDecisionTreeClassifier()
        model.tree_ = {
            'feature': features.index('retry_log_is_uncorrectable_error'),
            'threshold': 1,
            'left': {
                'feature': features.index('parity_columnwise_F_max_pooling_bit_min_interval'),
                'threshold': 4,
                'left': {
                    'feature': features.index('parity_columnwise_F_max_pooling_bit_max_interval'),
                    'threshold': 4,
                    'left': {'label': 1},
                    'right': {'label': 0}
                },
                'right': {'label': 0}
            },
            'right': {'label': 0}
        }
    elif method_name == "time_point_dq_beat_predictor":
        model = DQBeatPredictor()
    elif method_name == "time_point_risky_ce":
        model = RiskyCEClassifier()
    elif method_name in ["time_point_naive", "time_point_parity_ours", "time_point_gini", "time_point_lgb",
                         "time_point_xgb"]:

        train_data_path = "D:/competition_data/time_point/train_data/type_A"

        train_pos = read_dataframe(f'{train_data_path}/positive_train.csv')
        train_neg = read_dataframe(f'{train_data_path}/negative_train.csv')
        train_all = pd.concat([train_pos, train_neg])
        train_all['in_pos_parity_set'] = train_all['RetryRdErrLogParity'].apply(
            lambda x: 1 if process_hex(int(x)) in anomaly_parities else 0)

        train_all = train_all[features + ['label']]
        train_all["sn_name"] = train_all.index.get_level_values(0)
        train_all.drop_duplicates(keep="first", inplace=True)
        train_all.drop(columns=['sn_name'], inplace=True)

        if method_name == "time_point_gini":
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(train_all.drop(columns=['label']).values, train_all['label'].values)
        elif method_name == "time_point_lgb":
            model = LGBMClassifier()
            model.fit(train_all.drop(columns=['label']).values, train_all['label'].values)
        elif method_name == "time_point_xgb":
            model = XGBClassifier()
            model.fit(train_all.drop(columns=['label']).values, train_all['label'].values)
        else:
            train_all["sn_name"] = train_all.index.get_level_values(0)
            X, y = train_all.drop(columns=['label']).values, train_all['label'].values
            all_sns_set = set(train_all['sn_name'])
            pos_sn_set = set(train_all[train_all['label'] == 1]['sn_name'])

            model = CustomDecisionTreeClassifier(max_depth=max_depth,
                                                 first_layer_feature=first_layer_feature,
                                                 second_layer_feature=second_layer_feature)
            model.fit(X, y)

            print()

    os.makedirs("../results/stage_1/type_A", exist_ok=True)
    os.makedirs("../results/stage_2/type_A", exist_ok=True)
    os.makedirs("../results/stage_1/type_B", exist_ok=True)
    os.makedirs("../results/stage_2/type_B", exist_ok=True)

    test_stages = [1, 2]
    test_sn_types = ["type_A", "type_B"]
    for sn_type in test_sn_types:
        for stage in test_stages:
            test_data_path = os.path.join("D:/competition_data/time_point", f"test_stage_{stage}", sn_type)

            predict_result = predict(test_data_path, model)

            with open(f"../results/stage_{stage}/{sn_type}/{method_name}.pkl", "wb") as f:
                pickle.dump(predict_result, f)

    print()
