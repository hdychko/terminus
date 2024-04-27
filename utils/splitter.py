import logging
import numpy as np
import pandas as pd
import warnings
from scipy.stats import stats
from typing import Tuple
from typing import Sized
from typing import Union


class Splitter:

    @staticmethod
    def split_by_spearman_corr(value_col: pd.Series,
                                y_true: pd.Series,
                                n_bins_max: int,
                                n_bins_min: int = 2) -> pd.DataFrame:
        """
        Split an input column into equaly spaced bins in a way, that
        the number of observations per each bin has the largest monotonic dependence (the largest
        absolute value of Spearman's correlation) with y_true column.

        Columns `y_true` and `value_col` should be scaled before the input
        in a way, that all their values belong to [0, 1] interval

        :param value_col:           pd.Series, column needed a split
        :param y_true:              pd.Series, column based on which a split of `value_col` should be provided
        :param n_bins_max:          int, the maximum number of bins used for searching the best split
        :param n_bins_min:          int, the minimum number of bins used for searching the best split
        :return:                    pd.DataFrame, with the original input column to split and a split as the `bins` column
        """
        spearman_corr_list = []
        n_bins_list = []
        for n_bins in range(n_bins_min, n_bins_max + 1):
            df_bins = pd.DataFrame({'value_col': value_col, 'y_true': y_true, 'bins': pd.cut(y_true, n_bins)})

            df_bins_y_true = df_bins.groupby('bins').y_true.count().reset_index(name="y_true")
            df_bins_value_col = df_bins.groupby('bins').value_col.count().reset_index(name="value_col")

            df_bins_grouped = pd.merge(df_bins_y_true, df_bins_value_col, how="left", on="bins")
            # fill NaN with 0 for intervals, which do not have any values
            df_bins_grouped.y_true.fillna(0, inplace=True)
            df_bins_grouped.value_col.fillna(0, inplace=True)

            # in case all frequencies of bins are the same for both variables, the correlation is assigned to 1
            if (df_bins_grouped.value_col == df_bins_grouped.y_true).all():
                correlaation_val = 1
            else:
                # compute spearman correlation coefficient
                correlaation_val, p_val = stats.spearmanr(df_bins_grouped.value_col, df_bins_grouped.y_true)
            spearman_corr_list += [correlaation_val]
            n_bins_list += [n_bins]

        # find the number of bins with the largest absolute value of spearman correlation
        largest_corr_indx = np.nanargmax(np.abs(np.array(spearman_corr_list)))

        pearson_bins = pd.cut(value_col, n_bins_list[largest_corr_indx], retbins=True)[1]
        # adding +-inf
        pearson_bins = np.insert(pearson_bins, 0, -np.inf)
        pearson_bins = np.append(pearson_bins, np.inf)
        if value_col.name is None:
            col_name = "input_column"
        else:
            col_name = value_col.name
        return pd.DataFrame({col_name: value_col, 'bins': pd.cut(value_col, pearson_bins)})

    @staticmethod
    def split_by_uniform_distr(value_col: pd.Series, n_bins: int) -> pd.DataFrame:
        """
        Split an input column values into `n_bins` equal-width bins.
        The range of the input column is extended with +-infinity on corresponding edges to include the minimum and maximum its values.

        :param value_col:       pd.Series, input values
        :param n_bins:          int, number of equal-width bins
        :return:                pd.DataFrame, with the input column and its column name and bins with `bins` column name
        """
        uniform_bins = pd.cut(value_col, bins=n_bins, retbins=True)[1]
        # adding +-inf
        uniform_bins[0] = -np.inf
        uniform_bins[-1] = np.inf
        if value_col.name is None:
            col_name = "input_col"
        else:
            col_name = value_col.name
        return pd.DataFrame({col_name: value_col, 'bins': pd.cut(value_col, uniform_bins)})

    @staticmethod
    def split_by_quantiles(value_col: pd.Series, 
                           n_quantiles: int, 
                           precision: int = 3, 
                           retbins: bool=False) -> Union[pd.DataFrame, Tuple]:
        """
        Split column values into bins using quantiles
        :param value_col:           pd.Series, to split
        :param n_quantiles:         int, quantile order
        :return:                    pd.DataFrame, with initial column values and quantiles,
                                                  columns: [its column name, bins]
        """
        if value_col.nunique() == 1:
            raise ValueError("A numeric column should have more than one unique value to split into quantiles.")
        bins = pd.qcut(
            value_col, q=n_quantiles, duplicates='drop', precision=precision
        )
        if value_col.name is None:
            col_name = "input_col"
        else:
            col_name = value_col.name
        
        if retbins:
            bins_unique = bins.cat.categories
            return pd.DataFrame({col_name: value_col, 'bins': bins}), bins_unique
        else:
            return pd.DataFrame({col_name: value_col, 'bins': bins})

    @staticmethod
    def __generate_bin_plus_minus_epsilon_to_borders(left_bounds: pd.Series,
                                                     right_bounds: pd.Series, 
                                                     epsilon: float=1e-5):
        value_expected_bins_unique = pd.Series(name="bins")
        for i in range(0, len(left_bounds)):
            value_expected_bins_unique = pd.concat(
                (
                    value_expected_bins_unique,
                    pd.Series(
                        pd.Interval(
                            left_bounds[i], right_bounds[i],
                            closed="right"
                        ),
                        name="bins"
                    ),

                )
            )
        return value_expected_bins_unique

    def split_based_on_the_input_bins(self,
                                      value_col_actual: pd.Series,
                                      value_expected_bins: Union[pd.Series, pd.DataFrame],
                                      col_name: str,
                                      decimals: int = 3, 
                                      epsilon: float=1e-3) -> Tuple[pd.Series]:
        max_val_exp = np.array(list(map(lambda x: x.right, value_expected_bins))).max()
        max_val_act = value_col_actual.max()

        min_val_exp = np.array(list(map(lambda x: x.left, value_expected_bins))).min()
        min_val_act = value_col_actual.min()

        if ((round(max_val_exp, decimals) < round(max_val_act, decimals))) &  \
                (round(min_val_exp, decimals) < round(min_val_act, decimals)):
            msg = "[Splitter] Max of actual `{col_name}`={max_act}. Max of expected `{col_name}`={max_exp}".format(
                col_name=col_name, max_act=max_val_act, max_exp=max_val_exp
            )
            logging.warning(msg)
            warnings.warn(msg)

            value_missed_bins = Splitter.split_by_quantiles(pd.Series([max_val_exp, max_val_act]), n_quantiles=1, precision=decimals)

            value_expected_bins_unique = pd.IntervalIndex(
                pd.concat(
                    (pd.Series(value_expected_bins),
                     pd.Series(value_missed_bins.bins.unique()))
                ).unique()
            )

            value_expected_bins_unique = value_expected_bins_unique.sort_values()
            value_expected_bins_unique_left = value_expected_bins_unique.map(lambda x: x.left).astype(float).values
            value_expected_bins_unique_right = value_expected_bins_unique.map(lambda x: x.right).astype(float).values
            value_expected_bins_unique_right[-2] = value_expected_bins_unique_right[-1]

            value_expected_bins_unique_left = value_expected_bins_unique_left[:-1]
            value_expected_bins_unique_right = value_expected_bins_unique_right[:-1]
        elif (round(float(min_val_exp), decimals) > round(float(min_val_act), decimals)) & \
                (round(float(max_val_exp), decimals) > round(float(max_val_act), decimals)):
            msg = "[Splitter] Min of actual `{col_name}`={min_act}. Min of expected `{col_name}`={min_exp}".format(
                col_name=col_name, min_act=min_val_act, min_exp=min_val_exp
            )
            logging.warning(msg)
            warnings.warn(msg)

            value_missed_bins = Splitter.split_by_quantiles(pd.Series([min_val_act, min_val_exp]), n_quantiles=1, precision=decimals)

            value_expected_bins_unique = pd.IntervalIndex(
                pd.concat(
                    (pd.Series(value_expected_bins),
                     pd.Series(value_missed_bins.bins.unique())
                     )
                ).unique()
            )
            value_expected_bins_unique = value_expected_bins_unique.sort_values()
            value_expected_bins_unique_left = value_expected_bins_unique.map(lambda x: x.left).astype(float).values
            value_expected_bins_unique_right = value_expected_bins_unique.map(lambda x: x.right).astype(float).values

            value_expected_bins_unique_left[1] = value_expected_bins_unique_left[0]

            value_expected_bins_unique_left = value_expected_bins_unique_left[1:]
            value_expected_bins_unique_right = value_expected_bins_unique_right[1:]
        elif (round(max_val_exp, decimals) < round(max_val_act, decimals)) &\
                (round(min_val_exp, decimals) > round(min_val_act, decimals)):
            msg = "[Splitter] Actual `{col_name}`: min={min_act}; max={max_act}. " \
                  "Expected `{col_name}`: min={min_exp}; max={max_exp}".format(
                col_name=col_name, min_act=min_val_act, max_act=max_val_act, min_exp=min_val_exp, max_exp=max_val_exp
            )
            logging.warning(msg)
            warnings.warn(msg)
            value_missed_bins_max = Splitter.split_by_quantiles(pd.Series([max_val_exp, max_val_act]), n_quantiles=1, precision=decimals)
            value_missed_bins_min = Splitter.split_by_quantiles(pd.Series([min_val_act, min_val_exp]), n_quantiles=1, precision=decimals)
            value_expected_bins_unique = pd.IntervalIndex(
                pd.concat(
                    (
                        pd.Series(value_expected_bins),
                        pd.Series(value_missed_bins_max.bins.unique()),
                        pd.Series(value_missed_bins_min.bins.unique())
                    )
                ).unique()
            )
            value_expected_bins_unique = value_expected_bins_unique.sort_values()
            value_expected_bins_unique_left = value_expected_bins_unique.map(lambda x: x.left).astype(float).values
            value_expected_bins_unique_right = value_expected_bins_unique.map(lambda x: x.right).astype(float).values
            
            value_expected_bins_unique_left[-1] = value_expected_bins_unique_right[-2]
            value_expected_bins_unique_left = [round(int(elem * 10**decimals) / (10 ** decimals), decimals) for elem in value_expected_bins_unique_left]
            value_expected_bins_unique_right = [round(int(elem * 10**decimals) / (10 ** decimals), decimals) for elem in value_expected_bins_unique_right]
        elif (round(max_val_exp, decimals) >= round(max_val_act, decimals)) &\
                (round(min_val_exp, decimals) > round(min_val_act, decimals)):
            msg = "[Splitter] Actual `{col_name}`: min={min_act}. " \
                  "Expected `{col_name}`: min={min_exp}".format(
                col_name=col_name, min_act=min_val_act, min_exp=min_val_exp
            )
            logging.warning(msg)
            warnings.warn(msg)
            value_missed_bins_min = Splitter.split_by_quantiles(pd.Series([min_val_act, min_val_exp]), n_quantiles=1, precision=decimals)
            value_expected_bins_unique = pd.IntervalIndex(
                pd.concat(
                    (
                        pd.Series(value_expected_bins),
                        pd.Series(value_missed_bins_min.bins.unique())
                    )
                ).unique()
            )
            value_expected_bins_unique = value_expected_bins_unique.sort_values()
            value_expected_bins_unique_left = value_expected_bins_unique.map(lambda x: x.left).astype(float).values
            value_expected_bins_unique_right = value_expected_bins_unique.map(lambda x: x.right).astype(float).values
            
            value_expected_bins_unique_left[-1] = value_expected_bins_unique_right[-2]
            value_expected_bins_unique_left = [round(int(elem * 10**decimals) / (10 ** decimals), decimals) for elem in value_expected_bins_unique_left]
            value_expected_bins_unique_right = [round(int(elem * 10**decimals) / (10 ** decimals), decimals) for elem in value_expected_bins_unique_right]
        else:
            value_expected_bins_unique = value_expected_bins
            value_expected_bins_unique_left = value_expected_bins_unique.map(lambda x: x.left).astype(float).values
            value_expected_bins_unique_right = value_expected_bins_unique.map(lambda x: x.right).astype(float).values
            
        if (value_expected_bins_unique_left is not None) & (value_expected_bins_unique_right is not None):
            if epsilon is not None:
                value_expected_bins_unique_left[0] -= epsilon
                value_expected_bins_unique_right[-1] += epsilon
            value_expected_bins_unique = self.__generate_bin_plus_minus_epsilon_to_borders(
                value_expected_bins_unique_left,
                value_expected_bins_unique_right,
            )
        split_return, return_bins = pd.cut(
            value_col_actual, bins=pd.IntervalIndex(value_expected_bins_unique), retbins=True, precision=decimals
        )
        if split_return.isna().any():
            print(value_expected_bins_unique_left)
            print(value_expected_bins_unique_right)
            print(value_col_actual[split_return.isna()])
            raise ValueError("Incorrect Split")
        return split_return, return_bins

    def split_actual_values_into_bins_based_on_expected_split(self,
                                                              value_col_actual: pd.Series,
                                                              value_col_expected: pd.Series,
                                                              bins_method: str,
                                                              n_bins: int,
                                                              y_true_expected: pd.Series = None, decimals=3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split actual values in bins using binning of expected values.
        Supported binning methods: {spearman_corr, uniform, quantiles}

        :param value_col_actual:        pd.Series, gotten values (result of a recent experiment)
        :param value_col_expected:      pd.Series, expected values (historically based observations)
        :param bins_method:             str, kind of method which should be used to categorize
                                             actual and expected columns
        :param n_bins:                  int, number of bins used to make a column split in case `bins_method` = 'uniform',
                                             maximum number of bins used to make a 'spearman_corr' split
        :param y_true_expected:         pd.Series, optional column, which is used only
                                             when `bins_method` = 'uniform', and used as `y_true` parameter
        :return:                        (pd.DataFrame, pd.DataFrame), where
                                             - the first element of the returned tuple is
                                               split of actual data with columns: [actual_col_name, bins]
                                           - the second element of the returned tuple is
                                               split of expected data with columns: [expected_col_name, bins]
        """
        # split the variables into bins
        if value_col_actual.name is None:
            col_name = "input_col"
        else:
            col_name = value_col_actual.name

        value_actual_bins = pd.DataFrame(value_col_actual.copy(), columns=[col_name])
        if bins_method == "spearman_corr":
            if y_true_expected is None:
                msg = "[Metrics] Parameter `y_true_expected` should be specified in case `bins_method` is 'spearman_corr'"
                logging.error(msg)
                raise ValueError(msg)
            # split expected values into bins
            value_expected_bins = Splitter.split_by_spearman_corr(
                value_col=value_col_expected, y_true=y_true_expected, n_bins_max=n_bins
            )
            value_expected_bins_unique = pd.IntervalIndex(value_expected_bins.bins.unique())

            # split actual values based on expected binning
            value_actual_bins["bins"] = pd.cut(
                value_col_actual, bins=value_expected_bins_unique
            )

            value_expected_bins["bins"] = pd.IntervalIndex(value_expected_bins["bins"])
            value_actual_bins["bins"] = pd.IntervalIndex(value_actual_bins["bins"])
        elif bins_method == "uniform":
            # split expected values into bins
            value_expected_bins = Splitter.split_by_uniform_distr(
                value_col=value_col_expected, n_bins=n_bins
            )
            value_expected_bins_unique = pd.IntervalIndex(value_expected_bins.bins.unique())

            # split actual values based on expected binning
            value_actual_bins["bins"] = pd.cut(value_col_actual, bins=value_expected_bins_unique)

            value_expected_bins["bins"] = pd.\
                IntervalIndex(value_expected_bins["bins"])
            value_actual_bins["bins"] = pd.IntervalIndex(value_actual_bins["bins"])
        elif bins_method == "quantiles":
            # split expected values into bins
            value_expected_bins = Splitter.split_by_quantiles(value_col_expected, n_quantiles=n_bins, precision=decimals)

            value_actual_bins["bins"], value_actual_bins_ = self.split_based_on_the_input_bins(
                value_col_actual=value_col_actual,
                value_expected_bins=value_expected_bins,
                col_name=col_name,
                decimals=decimals
            )

            value_expected_bins["bins"] = self.split_based_on_the_input_bins(
                value_col_actual=value_col_expected,
                value_expected_bins=pd.DataFrame(value_actual_bins_, columns=["bins"]),
                col_name=col_name,
                decimals=decimals
            )[0]

            value_expected_bins["bins"] = pd.IntervalIndex(value_expected_bins["bins"])
            value_actual_bins["bins"] = pd.IntervalIndex(value_actual_bins["bins"])
        else:
            raise ValueError("[Metrics] Supported `bins_method` values: 'spearman_corr', 'uniform'")
        value_actual_bins.reset_index(drop=True, inplace=True)
        value_expected_bins.reset_index(drop=True, inplace=True)
        return value_actual_bins, value_expected_bins
