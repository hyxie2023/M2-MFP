import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import feather
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


os.makedirs(Config.train_data_path, exist_ok=True)
os.makedirs(Config.test_data_path, exist_ok=True)

ticket = pd.read_csv(Config.ticket_path)
pos_ticket = ticket[ticket['alarm_time'] <= 1717171200]  # 2024-06-01
pos_ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(pos_ticket['sn_name']), list(pos_ticket['alarm_time']))}
ticket_sn_map = {sn: sn_t for sn, sn_t in zip(list(ticket['sn_name']), list(ticket['alarm_time']))}

train_date_range = ("2024-01-01", "2024-06-01")
test_date_range = ("2024-06-01", "2024-10-01")


def datetime_to_timestamp(date: str) -> int:
    """
    Takes a date string in the format "YYYY-MM-DD" and returns the corresponding Unix timestamp.
    """

    return int(datetime.strptime(date, "%Y-%m-%d").timestamp())


def concat_in_chunks(chunks):
    chunks = [chunk for chunk in chunks if chunk is not None]
    if chunks:
        return pd.concat(chunks)
    return None


def parallel_concat(results, num_threads=16, chunk_size=200):
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    with Pool(num_threads) as pool:
        concatenated_chunks = pool.map(concat_in_chunks, chunks)

    return concat_in_chunks(concatenated_chunks)


def process_pos_file(sn_file):
    if pos_ticket_sn_map.get(sn_file[:-4]):
        end_time = pos_ticket_sn_map.get(sn_file[:-4])
        start_time = end_time - 30 * 24 * 3600

        data = feather.read_dataframe(os.path.join(Config.combined_sn_feature_data_path, sn_file))
        data = data[(data['LogTime'] <= end_time) & (data['LogTime'] >= start_time)]
        data = data.sort_values(by=['all_ce_log_num_15m'])[-20:]
        data['label'] = 1

        index_list = [(sn_file[:-4], log_time) for log_time in data['LogTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def process_neg_file(sn_file):
    if not pos_ticket_sn_map.get(sn_file[:-4]):
        end_time = 1717171200 - 30 * 24 * 3600
        # start_time = 1717171200 - 60 * 24 * 3600
        start_time = 1717171200 - 150 * 24 * 3600

        data = feather.read_dataframe(os.path.join(Config.combined_sn_feature_data_path, sn_file))
        data = data[(data['LogTime'] <= end_time) & (data['LogTime'] >= start_time)]
        if data.empty:
            return None
        data = data.sort_values(by=['all_ce_log_num_15m'])[-20:]
        data['label'] = 0

        index_list = [(sn_file[:-4], log_time) for log_time in data['LogTime']]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data
    return None


def get_positive_train_data():
    file_list = os.listdir(Config.combined_sn_feature_data_path)
    file_list = [x for x in file_list if x.endswith('.csv')]
    file_list.sort()

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_pos_file, file_list), total=len(file_list)))
    pos_data_all = parallel_concat(results)
    print(pos_data_all.info)

    feather.write_dataframe(pos_data_all, f'{Config.train_data_path}/positive_train.csv')


def get_negative_train_data():
    file_list = os.listdir(Config.combined_sn_feature_data_path)
    file_list = [x for x in file_list if x.endswith('.csv')]
    file_list.sort()

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_neg_file, file_list), total=len(file_list)))
    neg_data_all = parallel_concat(results)
    print(neg_data_all.info)

    feather.write_dataframe(neg_data_all, f'{Config.train_data_path}/negative_train.csv')


def get_test_data():
    file_list = os.listdir(Config.combined_sn_feature_data_path)
    file_list = [x for x in file_list if x.endswith('.csv')]
    file_list.sort()

    test_data_all = []
    sample_count_all = 0

    for file in tqdm(file_list):
        data_tmp = feather.read_dataframe(os.path.join(Config.combined_sn_feature_data_path, file))
        data_tmp = data_tmp[data_tmp['LogTime'] >= datetime_to_timestamp(test_date_range[0])]
        data_tmp = data_tmp[data_tmp['LogTime'] < datetime_to_timestamp(test_date_range[1])]
        if data_tmp.empty:
            continue

        index_list = [(file[:-4], log_time) for log_time in data_tmp['LogTime']]
        data_tmp.index = pd.MultiIndex.from_tuples(index_list)
        sample_count_all += len(data_tmp)
        test_data_all.append(data_tmp)

    test_data_all = parallel_concat(test_data_all)
    feather.write_dataframe(test_data_all, os.path.join(Config.test_data_path, "test_data.csv"))
    print()


if __name__ == "__main__":
    # get_positive_train_data()
    # get_negative_train_data()
    get_test_data()
