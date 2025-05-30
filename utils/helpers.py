import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read a file with either .csv or .feather extension.
    If the file extension is .csv, it first tries to read it as a CSV file.
    If reading as CSV fails with UnicodeDecodeError, it tries to read it as a Feather file.
    If the file extension is .feather, it reads it as a Feather file.
    If the file extension is neither .csv nor .feather, or if reading fails, it raises an exception.

    :param file_path: Path to the file
    :return: DataFrame containing the file data
    :raises: ValueError if the file extension is not supported or if reading fails
    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                return pd.read_feather(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read file as Feather after UnicodeDecodeError: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
    elif file_extension == '.feather':
        try:
            return pd.read_feather(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read Feather file: {e}")
    else:
        raise ValueError("Unsupported file extension. Only .csv and .feather are supported.")


def datetime_to_timestamp(date: str) -> int:
    """
    Takes a date string in the format "YYYY-MM-DD" and returns the corresponding Unix timestamp.
    """

    return int(datetime.strptime(date, "%Y-%m-%d").timestamp())


def merge_dicts(dict1: Dict[str, List], dict2: Dict[str, List]) -> Dict[str, List]:
    """
    Merges two dictionaries with string keys and list values.
    :param dict1: The first dictionary
    :param dict2: The second dictionary
    :return: Combined dictionary
    """
    return {k: dict1.get(k, []) + dict2.get(k, []) for k in set(dict1) | set(dict2)}


def plot_pr_curves(alg_performance_dict_list):
    plt.figure(figsize=(10, 8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('A3 P-R Curve')

    for i in alg_performance_dict_list:
        plt.plot(i['recall'], i['precision'],
                 marker='.', label=i['alg_name'])

    plt.legend()
    plt.show()
