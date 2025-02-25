import os
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def get_use_high_order_bit_level_features_names() -> List[str]:
    """
    Get the high order bit-level feature names. High order bit-level feature are the features that are calculated based
    on DQ-Beat Matrix.

    :return: A list of high order bit-level feature names.
    """
    high_order_bit_level_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for F_function in [
                "bit_count",
                "bit_min_interval",
                "bit_max_interval",
                "bit_max_consecutive_length",
                "bit_consecutive_length",
            ]:
                high_order_bit_level_feature_names.append(
                    f"dq_beat_{row_column}wise_{G_function}_{F_function}"
                )
    return high_order_bit_level_feature_names


features = get_use_high_order_bit_level_features_names()
train_all = pd.DataFrame()
all_sns_set = set(train_all["sn_name"])
pos_sn_set = set(train_all[train_all["label"] == 1]["sn_name"])


class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    """
    Custom decision tree classifier that uses a custom splitting criterion based on SN sets.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomDecisionTreeClassifier. Accepts any parameters supported by the parent
        DecisionTreeClassifier.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def _gini(y: List[int]) -> float:
        """
        Calculate the Gini impurity for a set of labels.

        :param y: Labels.
        :return: Gini impurity.
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    @staticmethod
    def _weighted_gini(y_left: List[int], y_right: List[int]) -> float:
        """
        Calculate the weighted Gini impurity for a binary split.

        :param y_left: Labels of the left split.
        :param y_right: Labels of the right split.
        :return: Weighted Gini impurity.
        """
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        return (n_left / n_total) * CustomDecisionTreeClassifier._gini(y_left) + (
                n_right / n_total
        ) * CustomDecisionTreeClassifier._gini(y_right)

    @staticmethod
    def _split_criterion(X: np.ndarray) -> Tuple[Optional[int], Optional[Any]]:
        """
        Determine the best split criterion for the data X. Iterates over all features (except the last column which is
        used for serial numbers) and all unique threshold values to find the split that minimizes the weighted Gini
        impurity.

        :param X: Data array where the last column contains serial numbers.
        :return: A tuple (feature_index, threshold) representing the best split.
                 Returns (None, None) if no valid split is found.
        """
        best_gini = float("inf")
        best_split = (None, None)
        n_samples, n_features = X.shape
        n_features -= 1  # The last column is assumed to be the serial number

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

                gini = CustomDecisionTreeClassifier._weighted_gini(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, threshold)

        return best_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomDecisionTreeClassifier":
        """
        Train the model.

        :param X: Training data.
        :param y: Training labels.
        :return: self
        """
        self.tree_ = self._build_tree(X, y)
        return self

    def _build_tree(
            self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively build the decision tree.

        Stopping conditions:
          1. If all samples belong to the same class.
          2. If the maximum depth is reached.
          3. If no valid split is found.

        :param X: Data array for the current node.
        :param y: Labels corresponding to X.
        :param depth: Current depth of the tree.
        :return: A dictionary representing the tree node.
        """
        # Stopping condition 1: If all samples belong to the same class
        if len(np.unique(y)) == 1:
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set),
                1: len(set(X[:, -1]) & pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Stopping condition 2: If maximum depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            if len(y) == 0:
                return {"label": 0, "class_counts": {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set),
                1: len(set(X[:, -1]) & pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Find the best split
        feature, threshold = CustomDecisionTreeClassifier._split_criterion(X)

        # Stopping condition 3: If no valid split is found
        if feature is None:
            if len(y) == 0:
                return {"label": 0, "class_counts": {0: 0, 1: 0}}
            label = len(set(X[:, -1]) & pos_sn_set) >= len(set(X[:, -1])) // 2
            class_counts = {
                0: len(set(X[:, -1])) - len(set(X[:, -1]) & pos_sn_set),
                1: len(set(X[:, -1]) & pos_sn_set),
            }
            return {"label": label, "class_counts": class_counts}

        # Recursively build left and right subtrees
        left_mask = X[:, feature] == threshold
        left_sns_set = set(X[left_mask][:, -1])
        right_sns_set = all_sns_set - left_sns_set
        left_mask = np.array([x in left_sns_set for x in X[:, -1]])
        right_mask = ~left_mask
        left_mask = X[:, feature] == threshold

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given samples.

        :param X: Data array of samples.
        :return: An array of predicted labels.
        """
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, tree: Dict[str, Any]) -> int:
        """
        Predict the label for a single sample.

        :param x: A single sample.
        :param tree: The current node of the decision tree.
        :return: The predicted label.
        """
        if "label" in tree:  # Leaf node
            return tree["label"]
        feature, threshold = tree["feature"], tree["threshold"]
        if (
                x[feature] == threshold
        ):  # If the sample's feature value equals the threshold
            return self._predict_one(x, tree["left"])
        else:  # If the sample's feature value is not equal to the threshold
            return self._predict_one(x, tree["right"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability estimates for each sample.

        :param X: Data array of samples.
        :return: An array of probability distributions for each sample.
        """
        return np.array([self._predict_proba_one(x, self.tree_) for x in X])

    def _predict_proba_one(self, x: np.ndarray, tree: Dict[str, Any]) -> List[float]:
        """
        Return the probability distribution for a single sample.

        :param x: A single sample.
        :param tree: The current node of the decision tree.
        :return: A list with the probability for each class.
        """
        if "class_counts" in tree:  # Leaf node
            class_counts = tree["class_counts"]
            total_samples = sum(
                class_counts.values()
            )  # Total number of samples in the leaf
            proba = [
                class_counts[0] / total_samples,
                class_counts[1] / total_samples,
            ]  # Compute probabilities for each class
            return proba
        feature, threshold = tree["feature"], tree["threshold"]
        if x[feature] == threshold:
            return self._predict_proba_one(x, tree["left"])
        else:
            return self._predict_proba_one(x, tree["right"])

    def get_tree(self) -> Dict[str, Any]:
        """
        Get the decision tree with feature names instead of feature indices.

        :return: A dictionary representing the decision tree.
        """

        def replace_feature_names(node: Dict[str, Any]) -> Dict[str, Any]:
            if "label" in node:
                return node
            feature_name = features[node["feature"]]
            left_tree = replace_feature_names(node["left"])
            right_tree = replace_feature_names(node["right"])
            return {
                "feature": feature_name,
                "threshold": node["threshold"],
                "left": left_tree,
                "right": right_tree,
            }

        return replace_feature_names(self.tree_)

    def extract_rules(
            self,
            node: Optional[Dict[str, Any]] = None,
            current_rule: Optional[List[Tuple[str, str, Any]]] = None,
    ) -> Any:
        """
        Extract decision rules from the tree.

        :param node: Current node in the decision tree. If None, start from the root.
        :param current_rule: The list of conditions accumulated so far.
        :return: A list of decision rules (each rule is a list of conditions).
        """
        if node is None:
            node = self.tree_
        if current_rule is None:
            current_rule = []

        if "label" in node:
            if node["label"] == 1:
                return [current_rule]
            else:
                return []

        rules = []
        feature = features[node["feature"]]
        threshold = node["threshold"]

        left_rule = current_rule + [(feature, "==", threshold)]
        right_rule = current_rule + [(feature, "!=", threshold)]

        rules += self.extract_rules(node["left"], left_rule)
        rules += self.extract_rules(node["right"], right_rule)

        return rules


def predict(test_data_path: str, model: Any) -> Dict[Any, List[Tuple[datetime, float]]]:
    """
    Predict anomaly scores for test data files using the provided model.

    :param test_data_path: Path to the directory containing test data files.
    :param model: A trained model with a predict method.
    :return: A dictionary mapping each serial number (sn) to a list of tuples,
             each containing a timestamp and the predicted score.
    """
    result_all: Dict[Any, List[Tuple[datetime, float]]] = {}
    for test_file in sorted(
            os.listdir(test_data_path),
            key=lambda filename: int(filename.split("_")[-1].split(".")[0]),
    ):
        test_df = pd.read_feather(f"{test_data_path}/{test_file}")
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

    return result_all
