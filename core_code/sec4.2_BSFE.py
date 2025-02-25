import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
# Todo: Add the default value for the data_path field
class Config:
    data_path: str = field(default="TBD", init=False)


class BSFE_DQBeatMatrix:
    """
    Calculate the DQ-Beat Matrix features for the bit-level CE info.
    """

    # The shape of DQ-Beat Matrix, for DDR4 memory, can be represented as an 8-row by 4-column binary matrix
    DQBeatMatrix_ROW_COUNT = 8
    DQBeatMatrix_COLUMN_COUNT = 4

    def __init__(self):
        self.binary_string_map_row = self.get_binary_string_map(
            self.DQBeatMatrix_COLUMN_COUNT
        )
        self.binary_string_map_column = self.get_binary_string_map(
            self.DQBeatMatrix_ROW_COUNT
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
        Get the row features of the binary string array. We define `f_` as 1d_BSFE.

        :param binary_string_array: list of binary strings
        :return: list of row features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        # Reduction-then-Aggregation
        f_binary_string_array = [
            self.binary_string_map_row[row] for row in binary_string_array
        ]
        max_pooling_f_binary_string_array = [
            max(column) for column in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(column) for column in list(zip(*f_binary_string_array))
        ]

        # Aggregation-then-Reduction
        rowwise_or_aggregate = 0
        for row in binary_string_array:
            rowwise_or_aggregate |= int(row, 2)
        rowwise_or_aggregate = bin(rowwise_or_aggregate)[2:].zfill(
            self.DQBeatMatrix_COLUMN_COUNT
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
        Get the column features of the binary string array. We define `f_` as 1d_BSFE.

        :param binary_string_array: list of binary strings
        :return: list of column features, including:
            - max_pooling_f_binary_string_array: max pooling of the f_binary_string_array
            - sum_pooling_f_binary_string_array: sum pooling of the f_binary_string_array
            - f_max_pooling_binary_string_array: F function of the max pooling of the binary string array
        """

        # Reduction-then-Aggregation
        f_binary_string_array = [
            self.binary_string_map_column[column] for column in binary_string_array
        ]
        max_pooling_f_binary_string_array = [
            max(row) for row in list(zip(*f_binary_string_array))
        ]
        sum_pooling_f_binary_string_array = [
            sum(row) for row in list(zip(*f_binary_string_array))
        ]

        # Aggregation-then-Reduction
        columnwise_or_aggregate = 0
        for column in binary_string_array:
            columnwise_or_aggregate |= int(column, 2)
        columnwise_or_aggregate = bin(columnwise_or_aggregate)[2:].zfill(
            self.DQBeatMatrix_ROW_COUNT
        )
        f_max_pooling_binary_string_array = list(
            self.binary_string_map_column[columnwise_or_aggregate]
        )

        return (
                max_pooling_f_binary_string_array
                + sum_pooling_f_binary_string_array
                + f_max_pooling_binary_string_array
        )

    def get_high_order_bit_level_features(
            self, err_log_dq_beat_matrix: int
    ) -> List[int]:
        """
        Get the DQ-Beat Matrix features for the bit-level CE.

        :param err_log_dq_beat_matrix: error log DQ-Beat Matrix
        :return: list of DQ-Beat Matrix features, including:
            - row features: row features of the binary string array
            - column features: column features of the binary string array
        """

        binary_err_log_dq_beat_matrix = bin(err_log_dq_beat_matrix)[2:].zfill(
            self.DQBeatMatrix_ROW_COUNT * self.DQBeatMatrix_COLUMN_COUNT
        )

        binary_row_array = [
            binary_err_log_dq_beat_matrix[i: i + self.DQBeatMatrix_COLUMN_COUNT]
            for i in range(
                0,
                self.DQBeatMatrix_ROW_COUNT * self.DQBeatMatrix_COLUMN_COUNT,
                self.DQBeatMatrix_COLUMN_COUNT,
            )
        ]
        binary_column_array = [
            binary_err_log_dq_beat_matrix[i:: self.DQBeatMatrix_COLUMN_COUNT]
            for i in range(self.DQBeatMatrix_COLUMN_COUNT)
        ]
        row_features = self.get_row_features(binary_row_array)
        column_features = self.get_column_features(binary_column_array)

        return row_features + column_features

    @staticmethod
    def process_file(file: str) -> List[int]:
        """
        Process the file to get the unique CE.

        :param file: file name
        :return: list of unique error log DQ-Beat Matrix
        """

        data = pd.read_csv(os.path.join(Config.data_path, file))
        return data.RetryRdErrLogParity.dropna().astype(np.int64).unique().tolist()

    def process_dq_beat_matrix(self, dq_beat_matrix: int) -> Tuple[int, List[int]]:
        """
        Process the dq_beat_matrix to get the high-order bit-level features.

        :param dq_beat_matrix: error log dq_beat_matrix
        :return: tuple of dq_beat_matrix and dq_beat_matrix features
        """
        return dq_beat_matrix, self.get_high_order_bit_level_features(dq_beat_matrix)

    def get_high_order_bit_level_features_dict(self) -> Dict[int, List[int]]:
        """
        Get the dq_beat_matrix dictionary.

        :return: dictionary of high-order bit-level features
        """
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_file, os.listdir(Config.data_path)),
                    total=len(os.listdir(Config.data_path)),
                )
            )
        dq_beat_matrix_set = set()
        for i in tqdm(results):
            dq_beat_matrix_set.update(i)
        dq_beat_matrix_set = sorted(list(dq_beat_matrix_set))

        _high_order_bit_level_features_dict = dict()
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_dq_beat_matrix, dq_beat_matrix_set),
                    total=len(dq_beat_matrix_set),
                )
            )
        for i, _high_order_bit_level_features in results:
            _high_order_bit_level_features_dict[i] = _high_order_bit_level_features

        return _high_order_bit_level_features_dict
