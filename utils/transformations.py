import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform_categorical_to_ohe_feature(col: pd.Series,
                                         values_to_ignore: list = [],
                                         if_alias_infront: bool = True,
                                         sep: str = "_.",
                                         col_alias: str = "") -> pd.DataFrame:
    """
    Transform an input column with categorical values into one hot encoded features using OneHotEncoder from sklearn.
    If values to ignore are not empty, these values are not considered while feature transformation.
    :rtype: object
    :param col:                 pd.Series, column with categories, which should be converted via one hoe encoding
                                           technique.
    :param values_to_ignore:    list, categories, which shouldn't be transformed.
    :param if_alias_infront:    str, if column name or an input alias should be placed infront of column's name or
                                     behind a category's name.
    :param sep:                 str, symbol of separation between alias (column name) and category's name.
                                     Empty string is used if nothing is provided.
    :param col_alias:           str, column's alias which should be used as part of output columns' names.
                                     The input column's name is used if nothing is provided.
    :return:                    pd.DataFrame, one hot encoded column
    """
    dummy_col_name = "0" * 10
    if col_alias == "":
        col_alias = col.name
    col_copy = copy.deepcopy(col)
    if values_to_ignore:
        col_copy.replace(values_to_ignore, dummy_col_name, inplace=True)
    enc = OneHotEncoder(handle_unknown='ignore')
    transformed_col = enc.fit_transform(pd.DataFrame(col_copy))
    if if_alias_infront:
        col_names = [col_alias + sep + str(elem) if elem != dummy_col_name else elem for elem in enc.categories_[0]]
    else:
        col_names = [str(elem) + sep + col_alias if elem != dummy_col_name else elem for elem in enc.categories_[0]]
    transformed_col = pd.DataFrame(transformed_col.toarray(), columns=col_names, dtype=int)
    if values_to_ignore:
        del transformed_col[dummy_col_name]
    return transformed_col


def transform_continuous_to_ohe_feature(col: pd.Series,
                                        cut_points: list,
                                        cut_labels: list,
                                        drop_nan: bool = True) -> pd.DataFrame:
    """
    Transform an input column with continuous values into one hot encoded features using OneHotEncoder from sklearn and
    input cut points.
    :rtype: object
    :param col:                 pd.Series, column with continuous values, which should be converted via one hoe encoding
                                           technique.
    :param cut_points:    list, points used to form intervals, which are used as categories for one hot encoder
    :param cut_labels:    list, names of intervals
    :return:                    pd.DataFrame, one hot encoded column
    """
    dummy_value = "0" * 10
    transformed_col = pd.cut(col, bins=cut_points, labels=cut_labels, right=False)
    if drop_nan:
        transformed_col = transformed_col.cat.add_categories(dummy_value)
        transformed_col.fillna(dummy_value, inplace=True)
    enc = OneHotEncoder(handle_unknown='ignore', categories=[transformed_col.cat.categories.tolist()])
    transformed_col = enc.fit_transform(pd.DataFrame(transformed_col))
    transformed_col = pd.DataFrame(transformed_col.toarray(), columns=enc.categories_, dtype=int)
    del transformed_col[dummy_value]
    transformed_col.columns = transformed_col.columns.get_level_values(0)
    return transformed_col
