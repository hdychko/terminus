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


def extract_col_With_the_highest_iv_among_correlated_features(corr_matrix: pd.DataFrame,
                                                              cut_off: float,
                                                              df_info: pd.DataFrame
                                                             ) -> Tuple[str, List[str]]:
    candidates = corr_matrix[corr_matrix.abs() >= cut_off].index.tolist()
    woe_candidates = [c for c in candidates if '_WoE' in c]
    numeric_candidates = [c for c in candidates if '_WoE' not in c]

    candidates_names = df_info.loc[
        [elem[:-4] for elem in woe_candidates] + numeric_candidates, 
        'IV'
    ].index.tolist()
    candidates_values = df_info.loc[
        [elem[:-4] for elem in woe_candidates] + numeric_candidates, 
        'IV'
    ].values

    cand_indx = candidates_values.argmax()
    return candidates_names[cand_indx], candidates_names
    
    
def mark_selected_canidate(df_info: pd.DataFrame, candidate: str) -> pd.DataFrame:
    df_info.loc[candidate, 'AsFeature'] = '+' \
        if (df_info.loc[candidate, 'AsFeature'] == '+') | \
            ((df_info.loc[candidate, 'AsFeature'] is np.nan)) \
        else '-'

    df_info.loc[candidate, 'Comment'] = df_info.loc[candidate, 'Comment'] + (
        '; Correlation checked;' \
            if ('; Correlation checked;' not in df_info.loc[candidate, 'Comment']) & \
                ('|Corr(' not in df_info.loc[candidate, 'Comment'])
            else ''
    )
    return df_info


def mark_correlated_features(features: List[str], 
                             df_info: pd.DataFrame,
                             feature_name: str, 
                             cut_off: float) -> pd.DataFrame:
    if len(features) > 0:
        for name in features:
            df_info.loc[name, 'AsFeature'] = '-'
            df_info.loc[name, 'Comment'] += f';  |Corr({feature_name}, {name})| >= {cut_off}'
    return df_info


def mark_correlated_features(df_stats: pd.DataFrame,
                             data: pd.DataFrame,
                             categorical_cols: List[str],
                             numeric_cols: List[str], 
                             cut_off: float
                            ):
    for col_name in categorical_cols + numeric_cols:
        if '|Corr(' in df_stats.loc[col_name, 'Comment']:
            continue

        col = col_name + '_WoE' if col_name in categorical_cols else col_name

        col_corr = data[
            [elem + '_WoE' for elem in categorical_cols] + numeric_cols
        ].corrwith(data[col], method='pearson', numeric_only=False)

        candidate, candidates_names = extract_col_With_the_highest_iv_among_correlated_features(
            corr_matrix=col_corr, cut_off=cut_off, df_info=df_stats
        )
        df_stats = mark_selected_canidate(df_stats, candidate)

        candidates_names.remove(candidate)
        df_stats = mark_correlated_features(candidates_names, df_stats, candidates_names, cut_off)
    return df_stats
