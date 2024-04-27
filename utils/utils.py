import copy

import numpy as np
import pandas as pd

from typing import List
from typing import Dict


def cramer_v(c):
    if isinstance(c, pd.DataFrame):
        c_matrix = c.to_numpy()
    else:
        c_matrix = c

    n = np.sum(c_matrix)
    r, k = c_matrix.shape

    c_row = c_matrix.sum(axis=1).reshape((-1, 1))
    c_col = c_matrix.sum(axis=0).reshape((1, -1))
    c_prod = c_row.dot(c_col) / n

    x2 = np.sum((c_matrix - c_prod) ** 2 / c_prod)
    v = np.sqrt(x2 / n / np.min((k - 1, r - 1)))

    return v


def cramer_v_bias_correction(c):
    if isinstance(c, pd.DataFrame):
        c_matrix = c.to_numpy()
    else:
        c_matrix = c

    n = np.sum(c_matrix)
    r, k = c_matrix.shape

    c_row = c_matrix.sum(axis=1).reshape((-1, 1))
    c_col = c_matrix.sum(axis=0).reshape((1, -1))
    c_prod = c_row.dot(c_col) / n

    phi2 = np.sum((c_matrix - c_prod) ** 2 / c_prod) / n
    phi2 = np.max((0, phi2 - (k - 1) * (r - 1) / (n - 1)))

    r = r - (r - 1) ** 2 / (n - 1)
    k = k - (k - 1) ** 2 / (n - 1)

    v = np.sqrt(phi2 / np.min((k - 1, r - 1)))

    return v



def transform_data_frames_to_list_of_columns(df: pd.DataFrame, df_benchmark: pd.DataFrame,
                                             numeric_col_names: List[str],
                                             categorical_col_names: List[str],
                                             ignore_null_values: bool = True) -> List[Dict]:
    """
    Transform pandas dataframe into list of dictionaries with the folowing format:
    [
        {
            "col_name": column_name,
            "values": {"benchmark": col_values_as_pd_series, "selected_time_period": col_values_as_pd_series},
            "type": numeric_or_categorical
        }
    ]
    :param df:                      pd.DataFrame, with the necessary columns of selected time period;
    :param df_benchmark:            pd.DataFrame, with the necessary columns of the benchmark;
    :param numeric_col_names:       List[str], names of columns which have numeric data type;
    :param categorical_col_names:   List[str], names of columns which have categorical data type;
    :param ignore_null_values:      bool, whether null values should be ignored in both input datasets and
                                          all the columns;
    :return:                        List[Dict], transformed pd.DataFrame into list of dictionaries
                                    with pd.Series and additional meta information
    """
    def ignore_null_values_func(col: pd.Series, ignore: bool) -> pd.Series:
        """
        Remove null values from the input pd.Series or all the values depending on the input flag
        :param col:     pd.Series, input column;
        :param ignore:  bool, whether null values should be excluded;
        :return:        pd.Series, with or without Nulls depending on the input parameter
        """
        return_value = copy.copy(col)
        if ignore:
             return_value = return_value[~return_value.isnull()].reset_index(drop=True)
        return return_value

    list_of_numeric_columns = [
        {
            "col_name": col,
            "values": {
                "benchmark": ignore_null_values_func(df_benchmark[col], ignore_null_values),
                "selected_time_period": ignore_null_values_func(df[col], ignore_null_values)
            },
            "type": "numeric"
        }
        for col in numeric_col_names
    ]
    list_of_categorical_columns = [
        {
            "col_name": col,
            "values": {
                "benchmark": ignore_null_values_func(df_benchmark[col], ignore_null_values),
                "selected_time_period": ignore_null_values_func(df[col], ignore_null_values)
            },
            "type": "categorical"
        }
        for col in categorical_col_names
    ]
    return list_of_numeric_columns + list_of_categorical_columns