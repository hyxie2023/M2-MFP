import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, NoReturn

import feather
import numpy as np
import pandas as pd

TIME_WINDOW_SIZE_MAP = {15 * 60: "15m", 60 * 60: "1h", 6 * 3600: "6h"}
TIME_RELATED_LIST = [15 * 60, 60 * 60, 6 * 3600]
FEATURE_EXTRACTION_INTERVAL = 15 * 60


# Todo: Add the default value for the Config class
@dataclass
class Config:
    """
    Configuration settings for file paths and parameters.

    Attributes:
        data_path (str): Path to the data directory.
        processed_time_point_data_path (str): Path to processed time point data.
        combined_sn_feature_data_path (str): Path to combined SN feature data.
        IMPUTE_VALUE (int): Value used for imputing missing data (default: -1).
    """

    data_path: str = field(default=r"TBD", init=False)
    processed_time_point_data_path: str = field(default=r"TBD", init=False)
    combined_sn_feature_data_path: str = field(default=r"TBD", init=False)
    IMPUTE_VALUE: int = -1


def unique_num_filtered(input_array: np.ndarray) -> int:
    """
    Deduplicate the input array, remove elements with the value IMPUTE_VALUE, and count the remaining unique elements.

    :param input_array: Input array
    :return: Number of unique elements after filtering
    """

    unique_array = np.unique(input_array)
    return len(unique_array) - int(Config.IMPUTE_VALUE in unique_array)


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


def get_max_sum_avg_values(
        input_values: pd.Series, valid_value_count: int
) -> Tuple[int, int, int]:
    """
    Get the max, sum, and average value of the input position.

    :param input_values: The input position.
    :param valid_value_count: The valid position count.
    """
    if valid_value_count == 0:
        return 0, 0, 0
    max_value = input_values.values.max()
    sum_value = input_values.values.sum()
    average_value = round(np.divide(sum_value, valid_value_count), 2)
    return max_value, sum_value, average_value


def calculate_ce_storm_count(log_times: pd.Series) -> int:
    """
    Calculate the number of CE storms.
    See https://github.com/hwcloud-RAS/SmartHW/blob/main/competition_starterkit/baseline_en.py for more details.

    CE storm definition:
    - Adjacent CE logs: If the time interval between two CE logs' LogTime is < 60s, they are considered adjacent logs.
    - If the number of adjacent logs exceeds 10, it is counted as one CE storm (note: if the number of adjacent logs
    continues to grow beyond 10, it is still counted as one CE storm).

    :param log_times: List of log LogTimes
    :return: Number of CE storms
    """

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


def get_aggregated_high_order_bit_level_features(
        window_df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Get the aggregated high order bit-level features.

    :param window_df: Data within the time window
    :return: Dictionary of aggregated high order bit-level features
    """
    aggregated_high_order_bit_level_features = dict()

    aggregated_high_order_bit_level_features["error_bit_count"] = window_df[
        "dq_beat_rowwise_sum_pooling_F_bit_count"
    ].values.sum()

    aggregated_high_order_bit_level_features["all_valid_err_log_count"] = window_df[
        "retry_log_is_valid"
    ].values.sum()

    columns_to_process = get_use_high_order_bit_level_features_names()

    for col in columns_to_process:
        max_col, sum_col, avg_col = get_max_sum_avg_values(
            window_df[col],
            aggregated_high_order_bit_level_features["all_valid_err_log_count"],
        )
        aggregated_high_order_bit_level_features[f"max_{col}"] = max_col
        aggregated_high_order_bit_level_features[f"sum_{col}"] = sum_col
        aggregated_high_order_bit_level_features[f"avg_{col}"] = avg_col

    dq_counts = dict()
    burst_counts = dict()
    for i in zip(
            window_df["dq_beat_rowwise_F_max_pooling_bit_count"].values,
            window_df["dq_beat_columnwise_F_max_pooling_bit_count"].values,
    ):
        dq_counts[i[0]] = dq_counts.get(i[0], 0) + 1
        burst_counts[i[1]] = burst_counts.get(i[1], 0) + 1

    for dq in [1, 2, 3, 4]:
        aggregated_high_order_bit_level_features[f"dq_count={dq}"] = dq_counts.get(
            dq, 0
        )
    for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
        aggregated_high_order_bit_level_features[
            f"burst_count={burst}"
        ] = burst_counts.get(burst, 0)

    aggregated_high_order_bit_level_features = {
        f"{key}": value
        for key, value in aggregated_high_order_bit_level_features.items()
    }

    return aggregated_high_order_bit_level_features


def get_spatio_features(window_df: pd.DataFrame) -> Dict[str, int]:
    """
    Extract spatial features including fault modes and counts of simultaneous row/column faults.
    See https://github.com/hwcloud-RAS/SmartHW/blob/main/competition_starterkit/baseline_en.py for more details.

    Fault mode definitions:
      - fault_mode_others: Other faults, where multiple devices exhibit faults.
      - fault_mode_device: Device faults, where multiple banks within the same device exhibit faults.
      - fault_mode_bank: Bank faults, where multiple rows within the same bank exhibit faults.
      - fault_mode_row: Row faults, where multiple cells in the same row exhibit faults.
      - fault_mode_column: Column faults, where multiple cells in the same column exhibit faults.
      - fault_mode_cell: Cell faults, where multiple cells with the same ID exhibit faults.
      - fault_row_num: Number of rows with simultaneous row faults.
      - fault_column_num: Number of columns with simultaneous column faults.

    :param window_df: Pandas DataFrame containing data within a specific time window.
    :return: A dictionary mapping spatial feature names to their integer values.
    """

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

    # Determine fault mode based on the number of faulty devices, banks, rows, columns, and cells
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

    # Record column address information for the same row
    row_pos_dict = {}
    # Record row address information for the same column
    col_pos_dict = {}

    for device_id, bank_id, row_id, column_id in zip(
            window_df["deviceID"].values,
            window_df["BankId"].values,
            window_df["RowId"].values,
            window_df["ColumnId"].values,
    ):
        current_row = "_".join([str(pos) for pos in [device_id, bank_id, row_id]])
        current_col = "_".join([str(pos) for pos in [device_id, bank_id, column_id]])
        row_pos_dict.setdefault(current_row, [])
        col_pos_dict.setdefault(current_col, [])
        row_pos_dict[current_row].append(column_id)
        col_pos_dict[current_col].append(row_id)

    for row in row_pos_dict:
        if unique_num_filtered(np.array(row_pos_dict[row])) > 1:
            spatio_features["fault_row_num"] += 1
    for col in col_pos_dict:
        if unique_num_filtered(np.array(col_pos_dict[col])) > 1:
            spatio_features["fault_column_num"] += 1

    return spatio_features


def process_single_sn(sn_file) -> NoReturn:
    """
    Process a single SN file to extract and aggregate features based on time windows,
    and then write the combined feature data to a feather file.

    :param sn_file: The filename of the SN file to process.
    """

    new_df = pd.read_feather(
        os.path.join(Config.processed_time_point_data_path, sn_file)
    )

    new_df["time_index"] = new_df["LogTime"] // FEATURE_EXTRACTION_INTERVAL
    new_df["CellId"] = (
            new_df["RowId"].astype(str) + "_" + new_df["ColumnId"].astype(str)
    )

    grouped = new_df.groupby("time_index")["LogTime"].max()
    window_end_time_list = grouped.tolist()

    combined_dict_list = []
    for end_time in window_end_time_list:
        combined_dict = {}
        window_df = new_df[
            (new_df["LogTime"] <= end_time)
            & (new_df["LogTime"] > end_time - 6 * 3600 - 15 * 60)
            ]
        combined_dict["ReportTime"] = window_df["LogTime"].max()
        window_df = window_df.drop_duplicates(
            subset=["deviceID", "BankId", "RowId", "ColumnId", "RetryRdErrLogParity"],
            keep="first",
        )
        combined_dict["LogTime"] = window_df["LogTime"].max()
        for time_window_size in TIME_RELATED_LIST[::-1]:
            end_logtime_of_filtered_window_df = window_df["LogTime"].max()
            window_df = window_df[
                window_df["LogTime"]
                >= end_logtime_of_filtered_window_df - time_window_size
                ]

            temporal_features = dict()
            temporal_features["read_ce_log_num"] = window_df[
                "error_type_is_READ_CE"
            ].values.sum()
            temporal_features["scrub_ce_log_num"] = window_df[
                "error_type_is_SCRUB_CE"
            ].values.sum()
            temporal_features["all_ce_log_num"] = len(window_df)
            temporal_features["log_happen_frequency"] = (
                time_window_size / temporal_features["all_ce_log_num"]
                if temporal_features["all_ce_log_num"]
                else 0
            )
            temporal_features["ce_storm_count"] = calculate_ce_storm_count(
                window_df["LogTime"]
            )

            aggregated_high_order_bit_level_features = (
                get_aggregated_high_order_bit_level_features(window_df)
            )
            spatio_features = get_spatio_features(window_df)

            combined_dict.update(
                {
                    f"{key}_{TIME_WINDOW_SIZE_MAP[time_window_size]}": value
                    for d in [
                        temporal_features,
                        spatio_features,
                        aggregated_high_order_bit_level_features,
                    ]
                    for key, value in d.items()
                }
            )
        combined_dict_list.append(combined_dict)
    combined_df = pd.DataFrame(combined_dict_list)
    feather.write_dataframe(
        combined_df, os.path.join(Config.combined_sn_feature_data_path, sn_file)
    )
