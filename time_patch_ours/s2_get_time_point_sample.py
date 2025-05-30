import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import feather
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


@dataclass
class Config:
    data_path: str = field(default=r"D:/m2mfp_data/sample_data_2000/type_A", init=False)
    processed_time_point_data_path: str = field(default=r"D:/m2mfp_data/time_point_data/type_A", init=False)
    ticket_path = r"D:/competition_data/competition_data_final/ticket.csv"


# parity有效校验编码
PARITY_VALID_MASK = 0x00000001
# 检验parity是否有UCE掩码
PARITY_UCE_MASK = 0x00000002

BINARY_PARITY_MATRIX_SHAPE = (8, 4)
PARITY_ROW_COUNT = BINARY_PARITY_MATRIX_SHAPE[0]
PARITY_COLUMN_COUNT = BINARY_PARITY_MATRIX_SHAPE[1]

DEFAULT_BINARY_STRING_ROW = "0" * PARITY_COLUMN_COUNT
DEFAULT_BINARY_STRING_COLUMN = "0" * PARITY_ROW_COUNT


def get_parity_feature_names():
    parity_feature_names = []
    for row_column in ["row", "column"]:
        for G_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for F_function in ["bit_count", "bit_min_interval", "bit_max_interval",
                               "bit_max_consecutive_length", "bit_consecutive_length"]:
                parity_feature_names.append(f"parity_{row_column}wise_{G_function}_{F_function}")
    return parity_feature_names


def get_manufacturer_info(manufacturer_series):
    manufacturer_array = manufacturer_series.fillna("").values
    return pd.DataFrame({
        "manufacturer_is_A": (manufacturer_array == "A").astype(int),
        "manufacturer_is_B": (manufacturer_array == "B").astype(int),
        "manufacturer_is_C": (manufacturer_array == "C").astype(int),
        "manufacturer_is_D": (manufacturer_array == "D").astype(int)
    })


def get_retry_log_info(retry_rd_err_log_series):
    retry_rd_err_log_array = retry_rd_err_log_series.fillna(0).replace('', 0).astype(int).values
    retry_log_is_valid = (retry_rd_err_log_array & PARITY_VALID_MASK) > 0
    retry_log_is_uncorrectable_error = ((retry_rd_err_log_array & PARITY_UCE_MASK) > 0) & retry_log_is_valid
    return pd.DataFrame({
        "retry_log_is_valid": retry_log_is_valid.astype(int),
        "retry_log_is_uncorrectable_error": retry_log_is_uncorrectable_error.astype(int)
    })


def get_parity_info(err_log_parity_series, parity_dict):
    parity_features = []

    for err_log_parity in err_log_parity_series:
        if err_log_parity in parity_dict:
            parity_features.append(parity_dict[err_log_parity])
        else:
            parity_features.append(parity_dict[0])  # Handle missing keys with a default list of zeros

    return pd.DataFrame(parity_features, columns=get_parity_feature_names())


def get_error_type_info(error_type_series):
    error_type_array = error_type_series.fillna("").values
    return pd.DataFrame({
        "error_type_is_CE": (error_type_array == "CE").astype(int),
        "error_type_is_READ_CE": (error_type_array == "CE.READ").astype(int),
        "error_type_is_SCRUB_CE": (error_type_array == "CE.SCRUB").astype(int)
    })


if __name__ == "__main__":
    os.makedirs(Config.processed_time_point_data_path, exist_ok=True)
    ticket = pd.read_csv(Config.ticket_path)
    ticket_sn = list(ticket['sn_name'])
    ticket_time = list(ticket['alarm_time'])
    ticket_sn_map = {sn: sn_t for sn, sn_t in zip(ticket_sn, ticket_time)}

    with open('parity_features.pkl', 'rb') as f:
        parity_dict = pickle.load(f)

    for i in tqdm(os.listdir(Config.data_path)):
        old_df = pd.read_csv(os.path.join(Config.data_path, i))
        if i[:-4] in ticket_sn:
            old_df = old_df[old_df['LogTime'] <= ticket_sn_map[i[:-4]]]
        old_df = old_df.sort_values(by='LogTime').reset_index(drop=True)
        new_df = old_df[["LogTime", "deviceID", "BankId", "RowId", "ColumnId", "MciAddr", "RetryRdErrLogParity"]].copy()
        new_df['deviceID'] = new_df['deviceID'].fillna(-1).astype(int)

        manufacturer_df = get_manufacturer_info(old_df["Manufacturer"])
        retry_log_df = get_retry_log_info(old_df["RetryRdErrLog"])
        error_type_df = get_error_type_info(old_df["error_type_full_name"])

        parity_dict_subset = {0: parity_dict[0]}
        for key in old_df["RetryRdErrLogParity"].unique():
            if not pd.isna(key):
                parity_dict_subset[key] = parity_dict[key]
        parity_features_df = get_parity_info(old_df["RetryRdErrLogParity"], parity_dict_subset)

        new_df = pd.concat(
            [new_df, manufacturer_df, retry_log_df, error_type_df, parity_features_df],
            axis=1)

        feather.write_dataframe(new_df, os.path.join(Config.processed_time_point_data_path, i))
