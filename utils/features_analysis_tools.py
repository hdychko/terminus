from typing import (
    Tuple, 
    List
)
import pandas as pd


def mark_features_with_not_null_level_lower_than(data: pd.DataFrame, not_null_level: float) -> pd.DataFrame:
    """
    If `%_not_nulls` column < `null_level`, 
    data['AsFeature'] = '-' and data['Comment'] is appended with '{not_null_level}% not null values;'
    
    :param data: pd.DataFrame, with `feature_name` as index and has columns: AsFeature, Comment, %_not_nulls;
    :param not_null_level: float, [0-100], for features with this % of null values or less, columns `AsFeature` and `Comment` are modified;
    
    :return: pd.DataFrame with modified columns
    """
    data.loc[data['%_not_nulls'] < not_null_level, 'AsFeature'] = '-'
    data.loc[data['%_not_nulls'] < not_null_level, 'Comment'] += f'; lower than {not_null_level}% not null values;'
    return data

