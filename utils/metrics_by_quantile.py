"""Class for cumulative metrics computation used in Streamlit Application, Notebooks.

#TODO: Add dtypes of pd.Series input arguments in model_monitoring_core
"""
import re
import warnings
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Iterable

from .splitter import Splitter
from .metrics import Metrics


class MetricsByQuantile(Metrics):
    def __init__(self,
                 metrics: List = None,
                 data: pd.DataFrame = None,
                 params_cols_match: dict = None):
        super().__init__(metrics=metrics, data=data, params_cols_match=params_cols_match)

    @staticmethod
    def cum_precision_by_quantile(cum_n_positive: pd.Series, cum_n_obs: pd.Series) -> pd.Series:
        """
        Cumulative precision = Cumulative # of positive class * 100 / Cumulative # of observations.
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        In case there are zeros in `cum_n_obs`, NaNs are returned
        :param cum_n_positive:      pd.Series, cumulative number of positive class by interval
        :param cum_n_obs:           pd.Series, cumulative number of observations by interval
        :return:                    pd.Series, cumulative precision value
        """
        cum_precision = cum_n_positive * 100 / cum_n_obs
        cum_precision.replace(np.inf, np.nan)
        cum_precision.name = "Cumulative Precision"
        return cum_precision

    @staticmethod
    def cum_recall_by_quantile(cum_n_positive: pd.Series, n_positive: int) -> pd.Series:
        """
        Cumulative recall = Cumulative # of positive class * 100 / total # of positive observations.
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        :param cum_n_positive:      pd.Series, cumulative number of positive class by interval
        :param n_positive:          int, total number of observations of positive class
        :return:                    pd.Series, cumulative recall
        """
        if (pd.Series([n_positive]).isnull().all()) | (not (pd.Series([n_positive]).astype(float) > 0).all()):
            msg = "[Metrics] Parameter `n_positive` should be int and strictly greater than 0"
            logging.warning(msg)
            warnings.warn(msg)
        cum_recall = cum_n_positive * 100 / n_positive
        cum_recall.replace(np.inf, np.nan, inplace=True)
        cum_recall.name = "Cumulative Recall"
        return cum_recall

    @staticmethod
    def cum_bad_rate_by_value(cum_val_positive: pd.Series,
                              cum_val: pd.Series) -> pd.Series:
        """
        Cumulative bad rate by value = Cumulative sum of column's values by positive class * 100 / Cumulative sum of values.
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        In case there are zeros in `cum_value`, NaNs are returned
        :param cum_val_positive:        pd.Series, cumulative sum of column's values by positive class by interval
        :param cum_val:             pd.Series, cumulative sum of column's values by interval
        :return:                       pd.Series
        """
        bad_rate_by_value = cum_val_positive * 100 / cum_val
        bad_rate_by_value.replace(np.inf, np.nan)
        bad_rate_by_value.name = "cum_bad_rate_by_value"
        return bad_rate_by_value

    @staticmethod
    def cum_fraud_value_detection_rate(sum_val_all: pd.Series, cum_val_positive: pd.Series) -> pd.Series:
        """
        Cumulative is the same as Fraud detection rate by quantile but operates with cumulative columns
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        In case there are zeros in `value`, NaNs are returned
        :param sum_val_all:              pd.Series, sum of column's values where predictions equal real and
                                                   belong to positive class by interval
        :param cum_val_positive:        pd.Series, sum of column's values of actual positive objects by interval
        :return:                        pd.Series
        """
        vd_rate = cum_val_positive * 100 / sum_val_all
        vd_rate.replace(np.inf, np.nan)
        vd_rate.name = "cum_fraud_value_detection_rate"
        return vd_rate

    @staticmethod
    def precision_by_quantile(positive_col: pd.Series, n_obs_col: pd.Series) -> pd.Series:
        """
        Precision by quantile = # of positive class * 100 / # of observations.
        Values are between 0 - 100.
        If no observations per quantile, NaN is returned.
        :param positive_col:        pd.Series, number of positive class per quantile
        :param n_obs_col:           pd.Series, number of observations per quantile
        :return:                    pd.Series, precision by quantile
        """
        precision = positive_col * 100 / n_obs_col
        precision.replace(np.inf, np.nan, inplace=True)
        precision.name = "Precision"
        return precision

    @staticmethod
    def recall_by_quantile(positive_col: pd.Series, n_positive: int) -> pd.Series:
        """
        Recall by quantile = # of positive class * 100 / # of all observations in positive class.
        Values are between 0 - 100.
        If number of all positive class observations is 0, an error is thrown.
        :param positive_col:        pd.Series, number of positive class per quantile
        :param n_positive:          pd.Series, number of observations per quantile
        :return:                    pd.Series, precision by quantile
        """
        if (pd.Series([n_positive]).isnull().all()) | (not (pd.Series([n_positive]).astype(float) > 0).all()):
            msg = "[Metrics] Parameter `n_positive` should be int and strictly greater than 0"
            logging.error(msg)
            # raise ValueError(msg)
        recall = positive_col * 100 / n_positive
        recall.name = "Recall"
        return recall

    @staticmethod
    def bad_rate_by_value_by_quantile(sum_val_positive: pd.Series,
                                      sum_val: pd.Series) -> pd.Series:
        """
        Bad rate by value = Sum of column's values by positive class * 100 / Sum of values.
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        In case there are zeros in `value`, NaNs are returned
        :param sum_val_positive:        pd.Series, sum of column's values by positive class by interval
        :param sum_val:                 pd.Series, sum of column's values by interval
        :return:                        pd.Series
        """
        bad_rate_by_value = sum_val_positive * 100 / sum_val
        bad_rate_by_value.replace(np.inf, np.nan)
        bad_rate_by_value.name = "bad_rate_by_value"
        return bad_rate_by_value

    @staticmethod
    def fraud_value_detection_rate_by_quantile(sum_val_all: pd.Series, sum_val_positive: pd.Series) -> pd.Series:
        """
        Fraud detection rate by quantile = Sum of column's values where predicted as positive belong to positive class * 100 / Sum of values by all positives.
        Values are between 0 - 100.
        Note, the input columns should be sorted from the highest to the lowest threshold value.
        In case there are zeros in `value`, NaNs are returned
        :param sum_val_all:              pd.Series, sum of column's values really belonging to positive class
        :param sum_val_positive:        pd.Series, sum of column's values of actual positive objects by interval
        :return:                        pd.Series
        """
        vd_rate = sum_val_positive * 100 / sum_val_all
        vd_rate.replace(np.inf, np.nan)
        vd_rate.name = "fraud_value_detection_rate"
        return vd_rate

    @staticmethod
    def f_beta_score_by_quantile(cum_positive: pd.Series, cum_n_obs: pd.Series, n_positive: int,
                                 beta: float = 1) -> float:
        """
        Compute f-score (weighted harmonic mean of precision and recall) based on the input quantiles:
        f_beta = (1 + beta ** 2) (precision * recall) / (precision * beta ** 2 + recall)

        :param cum_positive:        pd.Series, cumulative number of positive class by interval
        :param cum_n_obs:           pd.Series, cumulative number of observations by interval
                                           (value should be between 0 and 1)
        :param n_positive:          int, total number of observations of positive class
        :param beta:            float, weight of recall in the combined score;
        :return:                float, f-beta score value
        """
        cum_recall = cum_positive * 100 / n_positive
        cum_recall.replace(np.inf, np.nan, inplace=True)

        cum_precision = cum_positive * 100 / cum_n_obs
        cum_precision.replace(np.inf, np.nan)

        if (not cum_recall.isnull().all()) & (not cum_precision.isnull().all()):
            f_value = (1 + beta ** 2) * (cum_precision * cum_recall) / (cum_precision * beta ** 2 + cum_recall)
            f_value.name = "F-{} score".format(str(beta))
        else:
            msg = "[Metrics] Precision and/or recall value(s) equal(s) zero. NaN is returned"
            warnings.warn(msg)
            logging.warning(msg)
            f_value = pd.Series([np.nan] * len(cum_positive))
            f_value.name = "F-{} score".format(str(beta))
        return f_value

    @staticmethod
    def _general_info_by_quantiles(y_true: pd.Series,
                                   y_pred: pd.Series,
                                   y_quantiles: pd.Series,
                                   value_col: pd.Series = None,
                                   n_days: int = None) -> pd.DataFrame:
        """
        Compute statistics per quantile (quantiles are sorted in the descending order):
        - decile or quantile - the No of quantile or decile if n_quantiles = 10
        - quantiles - quantile's value
        - n_obs - number of observations
        - cum_n_obs - cumulative sum of `n_obs`
        - % of total population
        - Cumulative % of total population
        - n_positive - number of observations belonging to the positive class
        - cum_n_positive - cumulative sum of `n_positive`
        - scores - correspondence between quantiles and scores values
        - cum_val_positive - cumulative sum of `value_col` by real positive class
        - cum_val - cumulative sum of `value_col`

        :param y_true:          pd.Series, of target variable (note, only binary target should be used with classes:
                                           positive = 1, negative = 0)
        :param y_pred:          pd.Series, confidence of forecast
        :param value_col:       pd.Series, column to compute precision/recall using its values
        :param y_quantiles:     pd.Series, quantiles, to which each observation belong to
        :param n_days:          int, number of days to compute `Total per day`
        :return:                tuple, of pd.DataFrame, with quantiles and metrics per quantile
        """
        data = pd.DataFrame(
            {
                "y_pred": y_pred,
                "Quantile": y_quantiles,
                "value_col": value_col
            }
        )
        if y_true is not None:
            data["y_true"] = y_true
        data.reset_index(drop=True, inplace=True)

        # number of observations per quantile
        df_stats = data.groupby("Quantile").y_pred.count().reset_index(name="n_obs")

        # % of total population
        df_stats["% of total population"] = (df_stats["n_obs"] * 100 / data.shape[0]).round(2)

        # cumulative sum of `value_col`
        if value_col is not None:
            df_stats = pd.merge(
                df_stats,
                data.groupby("Quantile")["value_col"].sum().reset_index(name="sum_val"),
                on="Quantile", how="left"
            )
        if y_true is not None:
            # add number of true positive observations
            df_target_by_deciles = data.groupby(["Quantile"]).y_true.sum().reset_index(name="n_positive")

            if len(df_stats) != len(df_target_by_deciles):
                msg = f"[Metrics] Dimensions can't be matched: {len(df_stats)} and {len(df_target_by_deciles)}"
                logging.error(msg)
                raise ValueError(msg)
            if value_col is not None:
                df_target_by_deciles = pd.merge(
                    df_target_by_deciles,
                    data.groupby(["Quantile"]).apply(
                        lambda x: x[x["y_true"] == 1]["value_col"].sum()
                    ).reset_index(name="sum_val_positive"),
                    how="left", on="Quantile"
                )
        else:
            df_target_by_deciles = pd.DataFrame(data["Quantile"].unique(), columns=['Quantile'])

        df_stats = pd.merge(df_stats, df_target_by_deciles, how="outer", on="Quantile")
        df_stats['sortkey'] = df_stats.Quantile.map(lambda x: float(x.left))
        df_stats = df_stats.sort_values("sortkey", ascending=False)
        df_stats.reset_index(drop=True, inplace=True)

        if "n_positive" in df_stats.columns:
            df_stats["cum_n_positive"] = df_stats["n_positive"].cumsum()

            # Cumulative % of total population
            df_stats["Cumulative % of total population"] = df_stats["% of total population"].cumsum().round(2)
            df_stats["cum_n_obs"] = df_stats["n_obs"].cumsum()

        # cumulative `value_col`
        if value_col is not None:
            df_stats["cum_val"] = df_stats["sum_val"].cumsum()
            df_stats["cum_val_positive"] = df_stats["sum_val_positive"].cumsum()
            df_stats["sum_val_all"] = [data[data["y_true"] == 1]["value_col"].sum()] * df_stats.shape[0]

        if n_days is not None:
            df_stats["Total per day"] = df_stats["n_obs"] / n_days
            df_stats["Cumulative total per day"] = df_stats["cum_n_obs"] / n_days

        # col_name = "Decile" * (df_stats.shape[0] == 10) + "Quantile" * (df_stats.shape[0] != 10)
        df_stats["#Quantile"] = np.arange(df_stats.shape[0], 0, -1)
        return df_stats

    @staticmethod
    def remove_brackets_from_intervals(intervals: Iterable) -> List[str]:
        """
        Transform intervals from the format: '(value_1, value_2]' into intervals 'value_1-value_2'
        :param intervals:       Iterable, with intervals
        :return:                List, of string
        """
        # print(intervals)
        return [elem.split(", ")[0][1:] + "--" + elem.split(", ")[1][:-1] for elem in intervals.astype(str)]

    def _metrics_by_quantiles(self,
                              y_true: pd.Series,
                              y_pred: pd.Series,
                              y_quantiles: pd.Series,
                              list_of_metrics: List[str],
                              value_col: pd.Series = None,
                              n_days: int = None) -> pd.DataFrame:
        """
        Compute metrics provided in `list_of_metrics`
        Currently available metrics:
        - cum_precision_by_quantile
        - cum_recall_by_quantile
        - cum_bad_rate_by_value
        - f_beta_score_by_quantile
        - precision_by_quantile
        - recall_by_quantile

        :param y_true:                  pd.Series, of target variable (note, only binary target should be used with
                                                    classes: positive = 1, negative = 0)
        :param y_pred:                  pd.Series, confidence of forecast
        :param y_quantiles:             pd.Series, quantiles, to which each observation belong to
        :param value_col:               pd.Series, column to compute precision/recall per its values
        :param n_days:                  int, number of days to compute `Total per day`
        :param list_of_metrics:         list, of metrics which should be computed by quantile
        :return:                        pd.DataFrame, with columns:
                                                      ['Quantiles', '% of total population',
                                                       'Cumulative % of total population'] + list_of_metrics
        """
        value_col_name = value_col.name if value_col is not None else ""
        df_stats = self._general_info_by_quantiles(
            y_true=y_true, y_pred=y_pred, y_quantiles=y_quantiles, value_col=value_col, n_days=n_days
        )
        additional_params = dict()
        additional_cols = ["cum_val", "cum_val_positive", "sum_val", "sum_val_positive", 'sum_val_all']
        if all(col in df_stats.columns for col in additional_cols):
            additional_params = {col: df_stats[col] for col in additional_cols}
        positive_col = None
        n_positive = None
        cum_n_positive = None
        n_obs_col = None
        cum_n_obs = None
        if "n_positive" in df_stats.columns:
            positive_col = df_stats.n_positive
            n_positive = df_stats.n_positive.sum()
        if "cum_n_positive" in df_stats.columns:
            cum_n_positive = df_stats.cum_n_positive
        if "cum_n_obs" in df_stats.columns:
            cum_n_obs = df_stats["cum_n_obs"]
        if "n_obs" in df_stats.columns:
            n_obs_col = df_stats.n_obs
        # 'cum_positive' and 'cum_n_obs'
        df_stats_ = pd.DataFrame()
        if list_of_metrics != list():
            metrics_values = self._compute_metrics(
                metrics=list_of_metrics, positive_col=positive_col, n_positive=n_positive,
                cum_n_positive=cum_n_positive, cum_n_obs=cum_n_obs,
                n_obs_col=n_obs_col, **additional_params
            )
            df_stats_ = pd.concat(list(metrics_values.values()), axis=1)
        ll = [
                "% of total population", "Cumulative % of total population",
                "n_obs", "n_positive", "cum_n_positive",
                "Quantile", "#Quantile"
            ] + ["Cumulative total per day", "Total per day"] * (n_days is not None)
        df_stats = df_stats[[col for col in ll if col in df_stats.columns]]
        if not df_stats_.empty:
            df_stats = pd.concat([df_stats, df_stats_], axis=1)
        df_stats.rename(
            columns={
                "n_obs": "Total", "n_positive": "Bad", "cum_n_positive": "Cumulative Bad",
                "bad_rate_by_value": value_col_name + " Bad Rate",
                "cum_bad_rate_by_value": "Cumulative " + value_col_name + " Bad Rate ",
                "fraud_value_detection_rate": "% of total bads " + value_col_name,
                "cum_fraud_value_detection_rate": "Cumulative % of total bads " + value_col_name,
            },
            inplace=True
        )
        # df_stats.Quantile = df_stats.Quantile.astype(str)
        return df_stats

    def quantiles_analytics(self,
                            y_true: pd.Series,
                            y_pred: pd.Series,
                            list_of_metrics: List[str],
                            n_quantiles: int = 10,
                            value_col: pd.Series = None,
                            n_days: int = None) -> pd.DataFrame:
        """
        Compute input metrics per quantile.
        Return columns:
        - 'Scores' - `y_pred` values ranges per quantile
        - '% of total population' - percent of total population per quantile
        - 'Cumulative % of total population' - percent of total population per quantile (quantiles are sorted in
        descending order before computation)
        and input metrics

        :param y_true:                  pd.Series, of target variable (note, only binary target should be used with
                                                    classes: positive = 1, negative = 0)
        :param y_pred:                  pd.Series, confidence of forecast
        :param list_of_metrics:         list, of metrics which should be computed by quantile
        :param n_quantiles:             int, order of quantiles which should be used for split
        :param value_col:               pd.Series, column per which to compute bad rate
        :param n_days:                  int, number of days to compute `Total per day`
        :return:                        pd.DataFrame, with columns ['Scores', '% of total population',
                                        'Cumulative % of total population'] + aliases of the input metrics
        """
        y_split = Splitter().split_by_quantiles(y_pred, n_quantiles)
        # Compute metrics
        data = self._metrics_by_quantiles(
            y_true=y_true, y_pred=y_pred, y_quantiles=y_split["bins"],
            list_of_metrics=list_of_metrics, value_col=value_col, n_days=n_days
        )
        # change format of Quantiles from (a, b] to a-b
        data["Scores"] = self.remove_brackets_from_intervals(data["Quantile"])
        # drop indexes
        data.reset_index(drop=True, inplace=True)
        return data

    def _quantiles_analytics_by_benchmark_and_selected_period(self,
                                                             y_benchmark: dict,
                                                             y_slctd_period: dict,
                                                             list_of_metrics: List[str],
                                                             n_quantiles: int = 10,
                                                             decimals: int = 3,
                                                             **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute input metrics per quantile for benchmark data and for selected time period data.
        Return 2 data frames each with columns:
        - 'Decile' - No of a decile
        - 'Scores' - `y_pred` values ranges per quantile
        - '% of total population' - percent of total population per quantile
        - 'Cumulative % of total population' - percent of total population per quantile (quantiles are sorted in
        descending order before computation)
        and input metrics.
        If 'psi' metrics is provided, 'PSI' column will be included into returned data
        frame for selected time period data only.

        :param y_benchmark:         dict, of benchmark data with 'y_true' and 'y_pred' keys.
                                          - 'y_true' key should contain ground truth information of
                                          belonging to a positive or negative class (0, 1 values are expected)
                                          - 'y_pred' key should contain forecasting scores of
                                          belonging to a positive or negative class (values between 0 and 1 are expected)
                                          - 'value_col' contains column's values to compute precision/recall per this column
                                          - 'n_days' contains number of days to compute `Total per day`
        :param y_slctd_period:      dict, of data from a selected time period with the same keys as for benchmark data.
        :param list_of_metrics:     list, names of metrics, which should be computed
        :param n_quantiles:         int, order of quantile
        :param kwargs:              dict, with additional parameters needed for some
                                          metrics computations e.g. `zero_offset`
        :return:
        """
        # Split benchmark, selected time period into quantiles
        df_slctd_time_split, df_benchmark_split = Splitter().split_actual_values_into_bins_based_on_expected_split(
            value_col_actual=y_slctd_period["y_pred"], value_col_expected=y_benchmark["y_pred"],
            bins_method="quantiles", n_bins=n_quantiles, decimals=decimals
        )
        # Compute metrics for benchmark, selected time period

        df_stats_benchmark = self._metrics_by_quantiles(
            y_true=y_benchmark["y_true"], y_pred=y_benchmark["y_pred"], y_quantiles=df_benchmark_split["bins"],
            list_of_metrics=list(set(list_of_metrics) - {"psi"}),
            value_col=y_benchmark.get("value_col"), n_days=y_benchmark.get("n_days")
        )
        df_stats_slct_period = self._metrics_by_quantiles(
            y_true=y_slctd_period["y_true"], y_pred=y_slctd_period["y_pred"], y_quantiles=df_slctd_time_split["bins"],
            list_of_metrics=list(set(list_of_metrics) - {"psi"}),
            value_col=y_slctd_period.get("value_col"), n_days=y_slctd_period.get("n_days")
        )

        df_common = pd.merge(df_stats_benchmark, df_stats_slct_period, on="Quantile", how="left")
        # if col_bnch == col_slct:
        #     del df_common[col_bnch + "_y"]
        #     df_common.rename(columns={col_bnch + "_x": col_bnch}, inplace=True)

        # temp = df_common[~df_common[col_slct].isnull()][[col_bnch, col_slct]].to_dict(orient="list")
        # encod = dict(zip(temp[col_slct], temp[col_bnch]))
        # df_stats_slct_period[col_slct].replace(encod, inplace=True)

        mask = ~df_common["Quantile"].isnull()

        df_stats_slct_period.sort_values("#Quantile", ascending=False, inplace=True, ignore_index=True)
        df_stats_benchmark.sort_values("#Quantile", ascending=False, inplace=True, ignore_index=True)

        df_stats_benchmark_temp = df_stats_benchmark[mask].reset_index(drop=True)

        # psi y quantile, if it's specified
        if ("psi" in list_of_metrics) & (kwargs.get("zero_offset") is not None):
            df_stats_slct_period["PSI"] = self.psi(
                value_col_actual=df_stats_slct_period["% of total population"] / 100,
                value_col_expected=df_stats_benchmark_temp["% of total population"] / 100,
                zero_offset=kwargs["zero_offset"]
            )
        elif ("psi" in list_of_metrics) & (kwargs.get("zero_offset") is None):
            msg = "[Metrics] Parameter `zero_offset` should be provided for PSI computation."
            logging.error(msg)
            raise ValueError(msg)

        # reformat intervals view
        df_stats_benchmark["Scores"] = self.remove_brackets_from_intervals(df_stats_benchmark["Quantile"].astype(str))
        df_stats_slct_period["Scores"] = self.remove_brackets_from_intervals(df_stats_slct_period["Quantile"].astype(str))

        return df_stats_benchmark, df_stats_slct_period

    @staticmethod
    def quantiles_to_scores(df,
                            max_score_value,
                            min_score_value,
                            replace_negative_with_zero: bool = True,
                            scores_type="int",
                            direct_transformation=True,
                            decimals: int = 3):
        col_name = "Scores"
        split_score = df["Scores"].str.split("--")

        df["left_decile_value"] = split_score.apply(lambda x: float(x[0]))
        df["right_decile_value"] = split_score.apply(lambda x: float(x[1]))

        if replace_negative_with_zero:
            df.loc[df.left_decile_value < 0, "left_decile_value"] = 0.
            df.loc[df.right_decile_value < 0, "left_decile_value"] = 0.

        if direct_transformation:
            df["right_decile_value"] = df["right_decile_value"].values * (
                        max_score_value - min_score_value) + min_score_value
            df["left_decile_value"] = df["left_decile_value"].values * (
                        max_score_value - min_score_value) + min_score_value
        else:
            df["right_decile_value"] = max_score_value - df["right_decile_value"].values * (
                        max_score_value - min_score_value)
            df["left_decile_value"] = max_score_value - df["left_decile_value"].values * (
                        max_score_value - min_score_value)

        if scores_type == "int":
            df["right_decile_value"] = df["right_decile_value"].round(0).astype(int).astype(str)
            df["left_decile_value"] = df["left_decile_value"].round(0).astype(int).astype(str)
        elif scores_type == "float":
            df["right_decile_value"] = df["right_decile_value"].round(decimals).astype(float).astype(str)
            df["left_decile_value"] = df["left_decile_value"].round(decimals).astype(float).astype(str)

        if direct_transformation:
            df[col_name] = df["left_decile_value"] + "-" + df["right_decile_value"]
        else:
            df[col_name] = df["right_decile_value"] + "-" + df["left_decile_value"]
        del df["right_decile_value"], df["left_decile_value"]
        return df

    def quantiles_analytics_by_benchmark_and_selected_period(self,
                                                             y_benchmark: dict,
                                                             y_slctd_period: dict,
                                                             list_of_metrics: List[str],
                                                             max_score_value: float,
                                                             min_score_value: float,
                                                             scores_type: str,
                                                             direct_transformation: bool,
                                                             n_quantiles: int = 10,
                                                             zero_offset: float = 0.0001,
                                                             decimals: int = 3
                                                             ) -> pd.DataFrame:
        df_stats_benchmark, df_stats_slct_period = self._quantiles_analytics_by_benchmark_and_selected_period(
            y_benchmark=y_benchmark,
            y_slctd_period=y_slctd_period,
            list_of_metrics=list_of_metrics,
            n_quantiles=n_quantiles,
            zero_offset=zero_offset
        )

        df_stats_benchmark = self.quantiles_to_scores(
            df_stats_benchmark, max_score_value=max_score_value, min_score_value=min_score_value,
            scores_type=scores_type, direct_transformation=direct_transformation, decimals=decimals
        )

        df_stats_slct_period = self.quantiles_to_scores(
            df_stats_slct_period, max_score_value=max_score_value, min_score_value=min_score_value,
            scores_type=scores_type, direct_transformation=direct_transformation, decimals=decimals
        )

        # deciles psi
        df_deciles = pd.merge(
            df_stats_benchmark, df_stats_slct_period,
            how="outer", on="Quantile", suffixes=("_benchmark", "")
        )
        df_deciles.reset_index(drop=True, inplace=True)
        df_deciles.sort_values(
            "#Quantile_benchmark", ascending=False,
            inplace=True, ignore_index=True
        )
        df_deciles.rename(
            columns={
                "Scores": "Score Band",
                "Scores_benchmark": "Score Band_benchmark"
            }, inplace=True
        )

        df_deciles.rename(
            columns={
                "Precision": "Bad Rate",
                "Precision_benchmark": "Bad Rate_benchmark",

                "Cumulative Precision": "Cumulative Bad Rate",
                "Cumulative Precision_benchmark": "Cumulative Bad Rate_benchmark",

                "Recall": "% of Total Bads",
                "Recall_benchmark": "% of Total Bads_benchmark",

                "Cumulative Recall": "Cumulative % of Total Bads",
                "Cumulative Recall_benchmark": "Cumulative % of Total Bads_benchmark",

                "% of total population": "% of Total",
                "% of total population_benchmark": "% of Total_benchmark"
            }, inplace=True
        )
        df_deciles["Score Band"].fillna(df_deciles["Score Band_benchmark"], inplace=True)
        df_deciles["PSI"].fillna(1, inplace=True)
        for col in ["% of Total Bads", "Bad Rate", "Bad", "Total", "% of Total"]:
            df_deciles[col].fillna(0, inplace=True)

        columns = df_deciles.columns
        columns_from_benchmark = [col for col in columns if "benchmark" in col]
        columns_benchmark_rename_to = [re.sub("_benchmark", "", col) for col in columns_from_benchmark]
        df_deciles = pd.concat(
            (
                df_deciles[columns_from_benchmark].copy().rename(
                    columns=dict(zip(columns_from_benchmark, columns_benchmark_rename_to))
                ),
                df_deciles[[col for col in columns if "benchmark" not in col]].copy()
            ),
            keys=["Benchmark", "Selected Time Period"], axis=1
        )
        return df_deciles

    @staticmethod
    def compute_metrics_at_precision(df_deciles: pd.DataFrame,
                                    precision_val_list: list,
                                    list_of_metrics_to_output: list,
                                    alias: str = None) -> pd.DataFrame:
        """
        Find values of the specified metric by input values of precision
        :param df_deciles:                      pd.DataFrame, with deciles, cumulative precision, cumulative recall
                                                              per decile, etc.
        :param precision_val_list:              list, of precision's values to look for
        :param list_of_metrics_to_output:       list, of metrics' names which are in the input df_deciles data frame and
                                                      should be found by the input values of precision
        :param alias:                           str, alias to store the result
        :return:
        """
        df_recall_at_precision = pd.DataFrame()

        for prec_value in precision_val_list:
            argmin_indx = (df_deciles["Cumulative Precision"] - prec_value).abs().argmin()
            output_dict = dict.fromkeys(list_of_metrics_to_output)

            for metric_name in list_of_metrics_to_output:
                output_dict[metric_name] = df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index(metric_name)]

            output_dict["Scores"] = df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index("Scores")]
            output_dict["Cumulative % of total population"] = df_deciles.iloc[
                argmin_indx, df_deciles.columns.to_list().index("Cumulative % of total population")
            ]
            output_dict["Model"] = alias
            output_dict["Expected Precision"] = prec_value

            df_prec_recall_temp = pd.DataFrame.from_dict(output_dict, orient="index")
            df_prec_recall_temp = df_prec_recall_temp.T
            df_recall_at_precision = pd.concat((df_recall_at_precision, df_prec_recall_temp))
            df_recall_at_precision.reset_index(drop=True, inplace=True)
        return df_recall_at_precision
