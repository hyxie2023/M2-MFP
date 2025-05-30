import os
import pickle
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


@dataclass
class Config:
    data_path: str = field(default=r"D:/m2mfp_data/sample_data_2000/type_A", init=False)


class ParityFeature:
    """
    Calculate the parity features for the error log parity
    """

    # The shape of parity, for DDR4 memory, can be represented as an 8-row by 4-column binary matrix
    PARITY_ROW_COUNT = 8
    PARITY_COLUMN_COUNT = 4

    def __init__(self):
        self.binary_string_map_row = self.get_binary_string_map(
            self.PARITY_COLUMN_COUNT
        )
        self.binary_string_map_column = self.get_binary_string_map(
            self.PARITY_ROW_COUNT
        )

    def get_binary_string_features(
            self, binary_string: str
    ) -> Tuple[int, int, int, int, int]:
        """
        Get the binary string features, including features of a one-dimensional binary string.

        :param binary_string: binary string
        :return: tuple of binary string information, including:
            - bit_count: the number of valid bits in the binary string
            - bit_min_interval: the minimum interval between valid bits
            - bit_max_interval: the maximum interval between valid bits
            - bit_max_consecutive_length: the maximum length of consecutive valid bits
            - bit_consecutive_length: the cumulative continuous length of consecutive valid bits
        """

        bit_count = binary_string.count("1")
        bit_min_interval = len(binary_string)
        bit_max_interval = 0
        bit_max_consecutive_length = 0
        bit_consecutive_length = 0

        indices = self.indices_of_ones(binary_string)

        if len(indices) > 0:
            bit_max_interval = indices[-1] - indices[0]
            bit_max_consecutive_length = max([len(i) for i in binary_string.split("0")])
            bit_consecutive_length = binary_string.count("1") - sum(
                1 for i in binary_string.split("0") if len(i) > 0
            )
            if len(indices) > 1:
                bit_min_interval = min(np.diff(indices))

        return (
            bit_count,
            bit_min_interval,
            bit_max_interval,
            bit_max_consecutive_length,
            bit_consecutive_length,
        )

    def get_binary_string_map(
            self,
            binary_string_length: int,
    ) -> Dict[str, Tuple[int, int, int, int, int]]:
        """
        Convert the results of the function get_binary_string_info into a dictionary for convenient reuse.

        :param binary_string_length: the length of the binary string
        :return: dictionary of binary string information
        """

        binary_string_map = dict()

        for i in range(pow(2, binary_string_length)):
            binary_string = bin(i)[2:].zfill(binary_string_length)
            binary_string_info = self.get_binary_string_features(binary_string)
            binary_string_map[binary_string] = binary_string_info

        return binary_string_map

    @staticmethod
    def indices_of_ones(binary_string: str) -> List[int]:
        """
        Get the indices of the ones in a binary string.

        :param binary_string: binary string
        :return: list of indices of ones
        """

        return [index for index, char in enumerate(binary_string) if char == "1"]

    def get_row_features(self, binary_string_array: List[str]) -> List[int]:
        """
        Get the row features of the binary string array.

        :param binary_string_array: list of binary strings
        :return: list of row features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        f_binary_string_array = [
            self.binary_string_map_row[row] for row in binary_string_array
        ]
        max_pooling_f_binary_string_array = [
            max(column) for column in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(column) for column in list(zip(*f_binary_string_array))
        ]
        rowwise_or_aggregate = 0
        for row in binary_string_array:
            rowwise_or_aggregate |= int(row, 2)
        rowwise_or_aggregate = bin(rowwise_or_aggregate)[2:].zfill(
            self.PARITY_COLUMN_COUNT
        )
        f_max_pooling_binary_string_array = list(
            self.binary_string_map_row[rowwise_or_aggregate]
        )

        return (
                max_pooling_f_binary_string_array
                + sum_pooling_f_binary_string_array
                + f_max_pooling_binary_string_array
        )

    def get_column_features(self, binary_string_array: List[str]) -> List[int]:
        """
        Get the column features of the binary string array.

        :param binary_string_array: list of binary strings
        :return: list of column features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        f_binary_string_array = [
            self.binary_string_map_column[column] for column in binary_string_array
        ]
        max_pooling_f_binary_string_array = [
            max(row) for row in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(row) for row in list(zip(*f_binary_string_array))
        ]
        columnwise_or_aggregate = 0
        for column in binary_string_array:
            columnwise_or_aggregate |= int(column, 2)
        columnwise_or_aggregate = bin(columnwise_or_aggregate)[2:].zfill(
            self.PARITY_ROW_COUNT
        )
        f_max_pooling_binary_string_array = list(
            self.binary_string_map_column[columnwise_or_aggregate]
        )

        return (
                max_pooling_f_binary_string_array
                + sum_pooling_f_binary_string_array
                + f_max_pooling_binary_string_array
        )

    def get_parity_features(self, err_log_parity: int) -> List[int]:
        """
        Get the parity features for the error log parity.

        :param err_log_parity: error log parity
        :return: list of parity features, including:
            - row features: row features of the binary string array
            - column features: column features of the binary string array
        """

        binary_err_log_parity = bin(err_log_parity)[2:].zfill(
            self.PARITY_ROW_COUNT * self.PARITY_COLUMN_COUNT
        )

        binary_row_array = [
            binary_err_log_parity[i: i + self.PARITY_COLUMN_COUNT]
            for i in range(
                0,
                self.PARITY_ROW_COUNT * self.PARITY_COLUMN_COUNT,
                self.PARITY_COLUMN_COUNT,
            )
        ]
        binary_column_array = [
            binary_err_log_parity[i:: self.PARITY_COLUMN_COUNT]
            for i in range(self.PARITY_COLUMN_COUNT)
        ]
        row_features = self.get_row_features(binary_row_array)
        column_features = self.get_column_features(binary_column_array)

        return row_features + column_features

    @staticmethod
    def process_file(file: str) -> List[int]:
        """
        Process the file to get the unique error log parity.

        :param file: file name
        :return: list of unique error log parity
        """

        data = pd.read_csv(os.path.join(Config.data_path, file))
        return data.RetryRdErrLogParity.dropna().astype(np.int64).unique().tolist()

    def process_parity(self, parity: int) -> Tuple[int, List[int]]:
        """
        Process the parity to get the parity features.

        :param parity: error log parity
        :return: tuple of parity and parity features
        """
        return parity, self.get_parity_features(parity)

    def get_parity_dict(self) -> Dict[int, List[int]]:
        """
        Get the parity dictionary.

        :return: dictionary of parity features
        """
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_file, os.listdir(Config.data_path)),
                    total=len(os.listdir(Config.data_path)),
                )
            )
        parity_set = set()
        for i in tqdm(results):
            parity_set.update(i)
        parity_set = sorted(list(parity_set))

        parity_dict = dict()
        with Pool() as pool:
            results = list(
                tqdm(pool.imap(self.process_parity, parity_set), total=len(parity_set))
            )
        for i, parity_features in results:
            parity_dict[i] = parity_features

        return parity_dict


if __name__ == "__main__":
    parity_feature = ParityFeature()
    parity_dict = parity_feature.get_parity_dict()

    with open("parity_features.pkl", "wb") as pickle_file:
        pickle.dump(parity_dict, pickle_file)

    print()
