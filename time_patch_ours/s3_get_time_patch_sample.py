import os
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, List

import feather
import numpy as np
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
    ticket_path = r"D:\competition_data\competition_data_final\ticket.csv"


def get_bank_feature_names():
    bank_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for F_function in ["cell_count", "cell_max_interval", "cell_group_count", "cell_max_consecutive_length"]:
                bank_feature_names.append(f"bank_{row_column}wise_{G_function}_{F_function}")
    return bank_feature_names


def unique_num_filtered(input_array) -> int:
    """
    对输入的列表进行过滤, 去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

    :param input_array: 输入的列表
    :return: 返回经过过滤后的列表元素个数
    """
    unique_array = np.unique(input_array)
    return len(unique_array) - int(-1 in unique_array)


def get_kind_and_ptp(input_position: pd.Series, filter_num: int = "-1") -> Tuple[int, int]:
    """
    获取观测窗内 input_position 的种类数与极差
    """
    input_position = input_position[input_position != filter_num]
    if input_position.empty:
        return 0, 0
    unique_count = input_position.nunique()
    range_value = input_position.max() - input_position.min()
    return unique_count, range_value


def get_max_sum_average(input_position: pd.Series, valid_position_count: int) -> Tuple[int, int, int]:
    """
    获取观测窗内 input_position 的最大值与平均值
    """
    max_value = input_position.max()
    if valid_position_count == 0:
        return 0, 0, 0
    sum_value = input_position.values.sum()
    average_value = round(np.divide(sum_value, valid_position_count), 2)
    return max_value, sum_value, average_value


def get_group_count_and_max_consecutive_length(nums, m) -> Tuple[int, int]:
    if not nums:
        return 0, 0
    nums.sort()
    group_count, max_consecutive_length = 1, 1
    current_count = 1

    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] <= m:
            current_count += 1
        else:
            max_consecutive_length = max(max_consecutive_length, current_count)
            group_count += 1
            current_count = 1
    max_consecutive_length = max(max_consecutive_length, current_count)

    return group_count, max_consecutive_length


def get_1d_cell_features(cell_list: List[int], group_mode: str) -> Tuple[int, int, int, int]:
    assert group_mode in ["row", "column"], "group_mode must be either 'row' or 'column'"

    cell_count = unique_num_filtered(cell_list)
    cell_max_interval = max(cell_list) - min(cell_list) if cell_count > 1 else 0
    if group_mode == "row":
        cell_group_count, cell_max_consecutive_length = get_group_count_and_max_consecutive_length(cell_list, 1)
    else:
        cell_group_count, cell_max_consecutive_length = get_group_count_and_max_consecutive_length(cell_list, 8)

    return cell_count, cell_max_interval, cell_group_count, cell_max_consecutive_length


def get_row_features(grouped_column_ids) -> List[int]:
    grouped_column_ids_all = []
    row_features = []
    for column_ids in grouped_column_ids:
        cell_count, cell_max_interval, cell_group_count, cell_max_consecutive_length = \
            get_1d_cell_features(column_ids, "row")
        row_features.append([cell_count, cell_max_interval, cell_group_count, cell_max_consecutive_length])
        grouped_column_ids_all.extend(column_ids)
    grouped_column_ids_all = list(set(grouped_column_ids_all))
    max_pooling_F_binary_string_array = [max(column) for column in list(zip(*row_features))]
    sum_pooling_F_binary_string_array = [sum(column) for column in list(zip(*row_features))]
    F_max_pooling_binary_string_array = list(get_1d_cell_features(grouped_column_ids_all, "row"))

    return max_pooling_F_binary_string_array + sum_pooling_F_binary_string_array + F_max_pooling_binary_string_array


def get_column_features(grouped_row_ids) -> List[int]:
    grouped_row_ids_all = []
    column_features = []
    for row_ids in grouped_row_ids:
        cell_count, cell_max_interval, cell_group_count, cell_max_consecutive_length = \
            get_1d_cell_features(row_ids, "column")
        column_features.append([cell_count, cell_max_interval, cell_group_count, cell_max_consecutive_length])
        grouped_row_ids_all.extend(row_ids)
    grouped_row_ids_all = list(set(grouped_row_ids_all))
    max_pooling_F_binary_string_array = [max(row) for row in list(zip(*column_features))]
    sum_pooling_F_binary_string_array = [sum(row) for row in list(zip(*column_features))]
    F_max_pooling_binary_string_array = list(get_1d_cell_features(grouped_row_ids_all, "column"))

    return max_pooling_F_binary_string_array + sum_pooling_F_binary_string_array + F_max_pooling_binary_string_array


def calculate_ce_storm_count(log_times: pd.Series) -> int:
    ce_storm_interval_seconds = 60
    ce_storm_count_threshold = 10
    log_times = log_times.sort_values().reset_index(drop=True)
    ce_storm_count = 0
    consecutive_count = 0

    for i in range(1, len(log_times)):
        if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count > ce_storm_count_threshold:
            ce_storm_count += 1
            consecutive_count = 0

    return ce_storm_count


def get_use_parity_feature_names():
    parity_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for F_function in ["bit_count", "bit_min_interval", "bit_max_interval",
                               "bit_max_consecutive_length", "bit_consecutive_length"]:
                if G_function == "sum_pooling_F" and F_function != "bit_consecutive_length":
                    continue
                parity_feature_names.append(f"parity_{row_column}wise_{G_function}_{F_function}")
    return parity_feature_names


def get_err_parity_features(window_df, read_ce_only=True):
    err_parity_features = {}

    err_parity_features["error_bit_count"] = window_df['parity_rowwise_sum_pooling_F_bit_count'].values.sum()
    if read_ce_only:
        window_df = window_df[-100:]
        window_df = window_df[window_df["error_type_is_READ_CE"] == 1]

    err_parity_features["all_valid_err_log_parity_count"] = window_df['retry_log_is_valid'].values.sum()

    columns_to_process = get_use_parity_feature_names()

    for col in columns_to_process:
        max_col, sum_col, avg_col = get_max_sum_average(window_df[col],
                                                        err_parity_features["all_valid_err_log_parity_count"])
        err_parity_features[f"max_{col}"] = max_col
        err_parity_features[f"sum_{col}"] = sum_col
        err_parity_features[f"avg_{col}"] = avg_col

    dq_counts = dict()
    burst_counts = dict()
    for i in zip(window_df["parity_rowwise_F_max_pooling_bit_count"].values,
                 window_df["parity_columnwise_F_max_pooling_bit_count"].values):
        dq_counts[i[0]] = dq_counts.get(i[0], 0) + 1
        burst_counts[i[1]] = burst_counts.get(i[1], 0) + 1

    for dq in [1, 2, 3, 4]:
        err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)
    for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
        err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

    feature_suffix = "_read_ce" if read_ce_only else ""
    err_parity_features = {f"{key}{feature_suffix}": value for key, value in err_parity_features.items()}

    return err_parity_features


def get_spatio_features(window_df):
    spatio_features = {
        "fault_mode_others": 0,
        "fault_mode_device": 0,
        "fault_mode_bank": 0,
        "fault_mode_row": 0,
        "fault_mode_column": 0,
        "fault_mode_cell": 0,
        "fault_row_num": 0,
        "fault_column_num": 0,
    }

    if unique_num_filtered(window_df["deviceID"].values) > 1:
        spatio_features["fault_mode_others"] = 1
    elif unique_num_filtered(window_df["BankId"].values) > 1:
        spatio_features["fault_mode_device"] = 1
    elif (
            unique_num_filtered(window_df["ColumnId"].values) > 1
            and unique_num_filtered(window_df["RowId"].values) > 1
    ):
        spatio_features["fault_mode_bank"] = 1
    elif unique_num_filtered(window_df["ColumnId"].values) > 1:
        spatio_features["fault_mode_row"] = 1
    elif unique_num_filtered(window_df["RowId"].values) > 1:
        spatio_features["fault_mode_column"] = 1
    elif unique_num_filtered(window_df["CellId"].values) == 1:
        spatio_features["fault_mode_cell"] = 1

    grouped_row_ids = window_df.groupby(['BankId', 'ColumnId'])['RowId'].apply(list)
    for row_ids in grouped_row_ids:
        unique_num_filtered_row_ids = unique_num_filtered(set(row_ids))
        if unique_num_filtered_row_ids > 1:
            spatio_features["fault_column_mode_num"] += 1
    row_features = get_row_features(grouped_row_ids)
    grouped_column_ids = window_df.groupby(['BankId', 'RowId'])['ColumnId'].apply(list)
    for column_ids in grouped_column_ids:
        unique_num_filtered_column_ids = unique_num_filtered(set(column_ids))
        if unique_num_filtered_column_ids > 1:
            spatio_features["fault_row_mode_num"] += 1
    column_features = get_column_features(grouped_column_ids)

    for id, row_column_features in enumerate(row_features + column_features):
        spatio_features[get_bank_feature_names()[id]] = row_column_features

    spatio_features["mci_addr_kind"], spatio_features["mci_addr_ptp"] = get_kind_and_ptp(window_df['MciAddr'])
    spatio_features["device_kind"], spatio_features["device_ptp"] = get_kind_and_ptp(window_df['deviceID'])
    spatio_features["bank_kind"], spatio_features["bank_ptp"] = get_kind_and_ptp(window_df['BankId'])

    return spatio_features


def get_uce_pattern_features(window_df):
    uce_pattern_features = {}
    uce_pattern_features["retry_log_is_uncorrectable_error_count"] = window_df[
        'retry_log_is_uncorrectable_error'].values.sum()
    return uce_pattern_features


def process_single_sn(sn_file):
    new_df = feather.read_dataframe(os.path.join(Config.processed_time_point_data_path, sn_file))

    new_df['time_index'] = new_df['LogTime'] // 900
    new_df['CellId'] = new_df['RowId'].astype(str) + '_' + new_df['ColumnId'].astype(str)

    grouped = new_df.groupby('time_index')['LogTime'].max()
    window_end_time_list = grouped.tolist()

    combined_dict_list = []
    for end_time in window_end_time_list:
        combined_dict = {}
        window_df = new_df[(new_df['LogTime'] <= end_time) & (new_df['LogTime'] > end_time - 6 * 3600 - 15 * 60)]
        combined_dict["ReportTime"] = window_df["LogTime"].max()
        window_df = window_df.drop_duplicates(subset=['deviceID', 'BankId', 'RowId', 'ColumnId', 'RetryRdErrLogParity'],
                                              keep='first')
        combined_dict["LogTime"] = window_df["LogTime"].max()
        for time_window_size in TIME_RELATED_LIST[::-1]:
            end_logtime_of_filtered_window_df = window_df["LogTime"].max()
            window_df = window_df[window_df['LogTime'] >= end_logtime_of_filtered_window_df - time_window_size]

            temporal_features = {}
            temporal_features["read_ce_log_num"] = window_df['error_type_is_READ_CE'].values.sum()
            temporal_features["scrub_ce_log_num"] = window_df['error_type_is_SCRUB_CE'].values.sum()
            temporal_features["all_ce_log_num"] = len(window_df)
            temporal_features["log_happen_frequency"] = time_window_size / temporal_features["all_ce_log_num"] if \
                temporal_features["all_ce_log_num"] else 0
            temporal_features["ce_storm_count"] = calculate_ce_storm_count(window_df["LogTime"])

            err_parity_features = get_err_parity_features(window_df)
            spatio_features = get_spatio_features(window_df)
            uce_pattern_features = get_uce_pattern_features(window_df)

            combined_dict.update({f"{key}_{time_window_size_map[time_window_size]}": value for d in
                                  [temporal_features, spatio_features, err_parity_features, uce_pattern_features] for
                                  key, value in d.items()})
        combined_dict_list.append(combined_dict)
    combined_df = pd.DataFrame(combined_dict_list)
    feather.write_dataframe(combined_df, os.path.join(Config.combined_sn_feature_data_path, sn_file))


def subprocess_single_sn(args):
    sn_file_list = args[0]
    thread = args[1]

    for sn_file in tqdm(sn_file_list, f"thread: {thread}"):
        process_single_sn(sn_file)


TIME_RELATED_LIST = [15 * 60, 60 * 60, 6 * 3600]

if __name__ == "__main__":
    os.makedirs(Config.combined_sn_feature_data_path, exist_ok=True)

    # Single process
    # for i in tqdm(os.listdir("/home/zhoumin/data2/type_B/")):
    #     process_single_sn(i)

    # Multi process
    worker = 8
    threads = list(range(worker))

    sn_file_list = os.listdir(Config.processed_time_point_data_path)
    exist_sn_file_list = os.listdir(Config.combined_sn_feature_data_path)

    file_list_sp = [[] for _ in range(worker)]
    for i, file_name in enumerate([i for i in sn_file_list if i not in exist_sn_file_list and i.endswith("csv")]):
        index = i % worker
        file_list_sp[index].append(file_name)

    pool = Pool(worker)
    pool.map(subprocess_single_sn, zip(file_list_sp, threads))
    pool.close()
    pool.join()

    print()
