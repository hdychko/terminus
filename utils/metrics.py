"""Class for metrics computation used in Streamlit Application, Notebooks.

Metrics can be initialized to compute
- one particular metric (e.g. precision, recall, fraud_detection_rate, psi)
- set of metrics (provided as list with their names)

Metrics are split into those, that
 - depend on threshold value
 - do not dependent on it

#TODO: Add dtypes of pd.Series input arguments in model_monitoring_core
"""
import inspect
import warnings
import logging
import inspect
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from typing import Dict
from typing import Union
from typing import Any

from sklearn.metrics import auc, roc_curve
from typing import Tuple, List, Iterable

from .splitter import Splitter


class Metrics:

    def __init__(self,
                 metrics: List = None,
                 data: pd.DataFrame = None,
                 params_cols_match: dict = None):
        """
        In case several metrics are needed to compute, the following initialization should be provided
        :param metrics:             list, with metrics names provided as strings
        :param data:                pd.DataFrame, with data, based on which metrics should be computed
        :param params_cols_match:   dict, with correspondence:
                                          {'parameter name': 'column name in the provided dataset'}
        """
        self.metrics = metrics
        self.__bins = dict()
        if (params_cols_match is not None) & (data is not None):
            self.cols_params = {k: data[v] for k, v in params_cols_match.items()}
        elif (params_cols_match is None) & (data is not None):
            msg = "[Metrics] No `params_cols_match` was provided. Can't establish correspondance between "\
                  "`Metrics` methods parameters & data columns. "\
                  "The `data` parameter is ignored."
            logging.warning(msg)
            warnings.warn(msg)
        elif (params_cols_match is not None) & (data is None):
            msg = "[Metrics] No `data` was provided. Can't compute metrics values. " \
                  "The `params_cols_match` parameter is ignored."
            logging.warning(msg)
            warnings.warn(msg)
            self.cols_params = dict()
        else:
            self.cols_params = dict()

    @staticmethod
    def precision(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> float:
        """
        Compute a ratio of True Positive predictions among objects predicted as positive ones:
        precision = TP / (TP + FP)
        If there is no value predicted as positive class (TP + FP = 0), nan is returned.

        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float, precision value
        """
        y_pred = (y_prob >= threshold).astype(int)
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()
        false_positive = ((y_true == 0) & (y_pred == 1)).sum()
        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            msg = "[Metrics] No positive class as predicted objects were found while computing precision. NaN is returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def recall(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> float:
        """
        Compute a ratio of True Positive predictions among real positive objects:
        recall = TP / (TP + FN)
        If there is no positive class value (TP + FN = 0), nan is returned.

        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float, recall value
        """
        y_pred = (y_prob >= threshold).astype(int)
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()
        false_negative = ((y_true == 1) & (y_pred == 0)).sum()
        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            msg = "[Metrics] No positive class as input objects was found while computing recall. NaN will be returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def false_positive_rate(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> float:
        """
        Compute a ratio of misclassified real negative examples:
        false positive rate = FP / (FP + TN)
        In case no negative class was provided, nan is returned.

        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float, false positive rate value
        """
        y_pred = (y_prob >= threshold).astype(int)
        false_positive = ((y_true == 0) & (y_pred == 1)).sum()
        true_negative = ((y_true == 0) & (y_pred == 0)).sum()
        if (false_positive + true_negative) > 0:
            return false_positive / (false_positive + true_negative)
        else:
            msg = "[Metrics] No negative class as input objects were found while computing false_positive_rate."\
                  "NaN will be returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def true_positive_rate(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> float:
        """
        Compute a ratio of True Positive predictions among real positive objects:
        true positive rate = TP / (TP + FN)
        In case no positive class as input is provided, nan is returned.

        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float, true positive rate value
        """
        y_pred = (y_prob >= threshold).astype(int)
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()
        false_negative = ((y_true == 1) & (y_pred == 0)).sum()
        if (true_positive + false_negative) > 0:
            return true_positive / (true_positive + false_negative)
        else:
            msg = "[Metrics] No positive class as input objects was found while computing true_positive_rate."\
                  "NaN will be returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def fraud_value_detection_rate(value_col: pd.Series,
                                   y_true: pd.Series,
                                   y_prob: pd.Series,
                                   threshold: float) -> float:
        """
        Compute a percentage of saved values of correctly classified positive class:
        fraud value detection rate = sum of value_column for TP / (sum of value_column for actual positive class)

        In case no positive class as input objects is provided, nan is returned.

        :param value_col:       pd.Series, column that is taken to compute saved percentage
        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float
        """
        y_pred = (y_prob >= threshold).astype(int)
        true_positive_value = value_col[(y_true == 1) & (y_pred == 1)].sum()
        false_negative_value = value_col[(y_true == 1) & (y_pred == 0)].sum()
        if (true_positive_value + false_negative_value) > 0:
            return true_positive_value / (true_positive_value + false_negative_value)
        else:
            msg = "[Metrics] No positive class as input objects was found while computing fraud_value_detection_rate." \
                  "NaN will be returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def bad_rate_by_value(value_col: pd.Series,
                          y_true: pd.Series,
                          y_prob: pd.Series,
                          threshold: float) -> float:
        """
        Compute a percentage of values of correctly classified among all cases predicted ad positive class:
        bad rate value = sum of `value_column` for TP / (sum of `value_column` for all cases predicted as positive)

        In case no positive class predicted, nan is returned.

        :param value_col:       pd.Series, column that is taken to compute ratio
        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object being the positive class
                                           (value should be between 0 and 1)
        :param threshold:       float, value of the cut off point used to make predictions:
                                       - if `y_prob` < threshold, an object belongs to the negative class
                                       - if `y_prob` >= threshold, an object belongs to the positive class
        :return:                float
        """
        y_pred = (y_prob >= threshold).astype(int)
        true_positive_value = value_col[(y_true == 1) & (y_pred == 1)].sum()
        false_positive_value = value_col[(y_true == 0) & (y_pred == 1)].sum()
        if (true_positive_value + false_positive_value) > 0:
            return true_positive_value / (true_positive_value + false_positive_value)
        else:
            msg = "[Metrics] No predictions of positive class while computing `bad_rate_by_value`." \
                  "NaN will be returned"
            warnings.warn(msg)
            logging.warning(msg)
            return np.nan

    @staticmethod
    def positive_class_ratio(y_true: pd.Series) -> float:
        """
        Compute ratio of the positive class:

        #observations of positive class / # total observations

        :param y_true: pd.Series, with true positive class
        :return:       float, percentage of the positive class
        """
        return y_true.sum() / len(y_true)

    @staticmethod
    def gini(y_true: pd.Series, y_prob: pd.Series) -> float:
        """
        Gini Index, which is computed as 2*AUC - 1. Where AUC is area under the Receiver Operating Characteristic.
        Gini Index is between [-1 , 1], where -1 means bad model quality, 1 - perfect.
        :param y_true:          pd.Series, ground truth values
        :param y_prob:          pd.Series, predicted confidence/probability of object belonging to the positive class
        :return:                float, value of Gini Index
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_value = auc(fpr, tpr)
        gini_value = 2 * auc_value - 1
        return gini_value

    @staticmethod
    def psi_index_per_value(value_col_actual: pd.Series,
                            value_col_expected: pd.Series,
                            zero_offset: float) -> pd.Series:
        """
        Compute Population Stability Index for actual end expected input values using input zero offset.
        The standard formula is used for computations:
        PSI = [actual - expected] * ln[actual / expected]

        :param value_col_actual:        pd.Series, gotten values
        :param value_col_expected:      pd.Series, expected values
        :param zero_offset:             float, usually small value used for adding to
                                        input values to avoid taking logarithm from zero
        :return:                        pd.Series, psi values per input
        """
        # if all values of actual and expected are equaled to 0, than population is stable and 0 is returned
        if (value_col_actual.nunique() == 1) & (value_col_expected.nunique() == 1) & (value_col_expected.unique()[0] == 0):
            return pd.Series([0] * len(value_col_actual), name="psi")

        value_col_actual_copy = value_col_actual.copy()
        value_col_expected_copy = value_col_expected.copy()

        # replace zeros with the input offset to avoid logarithm uncertainty
        value_col_actual_copy.replace(0, zero_offset, inplace=True)
        value_col_expected_copy.replace(0, zero_offset, inplace=True)

        actual_expected_diff = value_col_actual_copy - value_col_expected_copy
        actual_expected_log_ratio = np.log(value_col_actual_copy/value_col_expected_copy)
        psi_value = (actual_expected_diff * actual_expected_log_ratio)
        psi_value.name = "psi"
        return psi_value

    def psi_total(self, value_col_actual: pd.Series, value_col_expected: pd.Series, zero_offset) -> float:
        """
        Compute Population Stability Index per input values and return it as a sum
        :param value_col_actual:        pd.Series, gotten values
        :param value_col_expected:      pd.Series, expected values
        :param zero_offset:             float, usually small value used for adding to
                                        input values to avoid taking logarithm from zero
        :return:                        float, sum of individual indexes
        """
        psi_indexes = self.psi_index_per_value(value_col_actual, value_col_expected, zero_offset)
        return psi_indexes.sum()

    def catch_metrics_with_param_name(self, param_name: str, list_of_metrics: List[str] = None) -> List:
        """
        Return all the public methods of the `Metrics` class (within input list of metrics)
        with the input parameter name
        :param param_name:              str, name of parameter that a metric has to have
        :param list_of_metrics:          List, of metrics which should be checked on having the input parameter name
        :return:                        List, of methods (among the input) with the input parameter name
        """
        methods = np.array([method for method in dir(self) if (method.startswith('_') is False) & callable(self.__getattribute__(method))])
        if list_of_metrics:
            metrics_by_param_mask = []
            # check whether each metric from the provided list is among `Metric`'s public methods and has a provided parameter
            for method in list_of_metrics:
                # check whether a metric is among `Metric`'s methods,
                # raise an error if not
                if method not in methods:
                    msg = "[Metrics] No method with name {}".format(method)
                    logging.error(msg)
                    raise ValueError(msg)
                # check whether metric's parameters contain a provided parameter
                else:
                    metrics_by_param_mask += [
                        param_name in set(inspect.getfullargspec(self.__getattribute__(method))[0])
                    ]
            metrics_by_param = list(np.array(list_of_metrics)[metrics_by_param_mask])
        else:
            # if no list is provided, each `Metric`'s public methods is checked on having a provided parameter
            metrics_by_param_mask = np.array(
                [
                    param_name in set(inspect.getfullargspec(self.__getattribute__(method))[0]) \
                    for method in methods if callable(self.__getattribute__(method))
                ]
            )
            metrics_by_param = list(methods[metrics_by_param_mask])
        return metrics_by_param

    def _compute_single_metric_by_name_and_thresholds_list(self,
                                                           metric: str,
                                                           thresholds_list: List[float],
                                                           decimal: int) -> dict:
        """
        Compute metric values by its name and for each of the threshold value provided in `thresholds_list` parameter
        :param metric:          str, metrics name
        :param thresholds_list: list, of thresholds to compute metric values
        :param decimal:         int, number of values after integer part
        :return:                dict, in format {'threshold': [...], `metric`: [...]}
        """
        # get parameter name and its value from self.cols_params
        params = self.cols_params.copy()
        metrics_value = []
        # compute metric value per provided threshold
        for p in thresholds_list:
            p = round(p, decimal)
            params.update(threshold=p)
            metrics_value += [self.__getattribute__(metric)(**params)]
        # form return as dictionary
        metrics_value = {"threshold": thresholds_list, metric: metrics_value}
        metrics_value[metric] = list(np.array(metrics_value[metric]).round(decimal))

        return metrics_value

    def _compute_metrics_per_thresholds(self, metrics: List[str], n_thresholds: int, decimal: int = 3) -> dict:
        """
        Compute metrics by their names using columns' matching dictionary
        for each of the generated thresholds
        :param metrics:           list, of metrics which should be computed for each threshold's value
        :param n_thresholds:      int, number of values equaly spaced between [0, 1]
        :param decimal:           int, number of values after integer part (default value is 3)
        :return:                  dict, with thresholds as keys and dictionary of metrics names and their values as values
                                  e.g. {0.0: {"precision": 1, "fraud_detection_rate": 0}, 0.1: {"precision": 1, "recall": 0}}
        """
        df_metrics = pd.DataFrame(
            np.arange(0, 1., step=1 / n_thresholds).round(decimal),
            columns=["threshold"]
        )
        for metrics_name in metrics:
            metrics_value = self._compute_single_metric_by_name_and_thresholds_list(
                metric=metrics_name, thresholds_list=df_metrics.threshold.to_list(), decimal=decimal
            )
            if len(metrics_value[metrics_name]) != len(df_metrics):
                raise ValueError(f"Dimension mismatch: {len(metrics_value)} != {len(df_metrics)}")
            metrics_value = pd.DataFrame.from_dict(metrics_value)
            df_metrics = pd.merge(df_metrics, metrics_value, how="inner", on="threshold")
        df_metrics.set_index("threshold", inplace=True)
        return df_metrics.to_dict(orient="index")

    def _compute_metrics(self, metrics: List[str], **kwargs) -> dict:
        """
        Compute each of the input metric by its name using input parameters
        If the metrics list contains name, which is not attribute nor method of `Metrics` class, an error is thrown
        :param metrics:             tuple, of metrics names
        :param kwargs:              dict, of parameters needed to compute metrics
        :return:                    dict, in format {"metric_name": metric_value, ... }
        """
        metrics_dict = dict.fromkeys(metrics)
        for metrics_name in metrics:
            # collect all the parameters needed by the method
            params = dict()
            # get parameters needed by the method
            metrics_args = set(inspect.getfullargspec(self.__getattribute__(metrics_name))[0]).difference({"self"})
            # update with **kwargs
            if metrics_args.intersection(set(kwargs.keys())) != {}:
                params.update({k: kwargs[k] for k in metrics_args.intersection(set(kwargs.keys()))})
            if metrics_args.intersection(set(self.cols_params.keys())) != {}:
                # update with data parameters
                params.update({k: self.cols_params.copy()[k] for k in metrics_args.intersection(set(self.cols_params.keys()))})
            # compute metric value
            metrics_dict[metrics_name] = self.__getattribute__(metrics_name)(**params)
        return metrics_dict

    def compute(self, n_thresholds: int = None, decimal: int = 3, **kwargs) -> dict:
        """
        Compute metrics listed within `Metrics` initialization and using corresponding parameters

         - if `n_thresholds` is provided as input, each metric from initialization is computed per each threshol value
         - if just `threshold` is provided, threshold dependent metrics are computed for this particular value
         - if both `n_thresholds` and`threshold` are provided, an error is thrown
         - if nothing is provided, but metrics with `threshold` parameter are provided
            while initialization, an error is thrown

        If no threshold dependent metric is provided, there is still "other" key
        in returned dictionary but with empty value.
        The same is applied to threshold dependent metrics and "threshold_dependent" key

        :param n_thresholds: int, number of values equaly spaced between [0, 1]
        :param decimal:      int, how many numbers after comma will be shown
        :param kwargs:       dict, additional parameters needed to compute metrics
        :return:             dict,
                             - in case `n_thresholds` parameter is provided, the format is:
                                   {"threshold_dependent":
                                        {0.1:
                                            {"metrics_name1": metrics_value1, "metrics_name2": metrics_value2},
                                         0.2:
                                            {"metrics_name1": metrics_value1, "metrics_name2": metrics_value2},
                                        ...
                                    "other": {"metrics_name": metrics_value, "metrics_name_other": metrics_value_other}}
                            - in case `threshold` parameter is provided, the format is:
                                   {"threshold_dependent":
                                        {"metrics_name1": metrics_value1, ..., "metrics_name2": metrics_value2}
                                    "other": {"metrics_name": metrics_value, "metrics_name_other": metrics_value_other}}

        """
        # define metrics that have `threshold` parameter & do not have
        metrics_by_thresholds = self.catch_metrics_with_param_name(param_name="threshold", list_of_metrics=self.metrics)
        metrics_no_thresholds = list(set(self.metrics).difference(set(metrics_by_thresholds)))

        # check which parameters are passed and which type of computations should be provided
        computed_metrics = dict.fromkeys(["threshold_dependent", "other"])
        # if n_thresholds should be generated and metric(s) computed per each value
        if (len(metrics_by_thresholds) > 0) & (n_thresholds is not None) & (kwargs.get("threshold") is None):
            computed_metrics["threshold_dependent"] = self._compute_metrics_per_thresholds(
                metrics=metrics_by_thresholds, n_thresholds=n_thresholds, decimal=decimal
            )
        # if just one `threshold` value is input and metric(s) should be computed using this particular value
        elif (len(metrics_by_thresholds) > 0) & (n_thresholds is None) & (kwargs.get("threshold") is not None):
            computed_metrics["threshold_dependent"] = self._compute_metrics(
                metrics_by_thresholds, **kwargs
            )
        # if both values `n_thresholds` and `threshold` are input, just an error is thrown
        elif (len(metrics_by_thresholds) > 0) & (n_thresholds is not None) & (kwargs.get("threshold") is not None):
            msg = "[Metrics] Only one of parameters: `n_threshold` or `threshold` should be provided"
            logging.error(msg)
            raise ValueError(msg)
        # if no `n_thresholds` nor `threshold` are input, just an error is thrown
        elif (len(metrics_by_thresholds) > 0) & (n_thresholds is None) & (kwargs.get("threshold") is None):
            msg = "[Metrics] Parameter `n_threshold` or `threshold` should not be None"
            logging.error(msg)
            raise ValueError(msg)
        # if `n_thresholds` or `threshold` is input and no threshold dependent metric is specified,
        # just a warning is thrown
        elif (len(metrics_by_thresholds) == 0) & ((n_thresholds is not None) | (kwargs.get("threshold") is not None)):
            msg = "[Metrics] Parameter `n_threshold` or `threshold` is not used since no threshold dependent metrics "\
                  "is in input"
            logging.warning(msg)
            warnings.warn(msg)
        # compute threshold independent metrics
        computed_metrics["other"] = dict()
        for metrics_name in metrics_no_thresholds:
            # collect all the parameters needed by the method
            params = dict()
            # get parameters needed by the method
            metrics_args = set(inspect.getfullargspec(self.__getattribute__(metrics_name))[0]).difference({"self"})
            # update with **kwargs
            if metrics_args.intersection(set(kwargs.keys())) != {}:
                params.update({k: kwargs[k] for k in metrics_args.intersection(set(kwargs.keys()))})
            # update with data columns
            if metrics_args.intersection(set(self.cols_params.keys())) != {}:
                params.update({k: self.cols_params[k] for k in metrics_args.intersection(set(self.cols_params.keys()))})
            computed_metrics["other"][metrics_name] = self.__getattribute__(metrics_name)(**params)
        return computed_metrics

    def make_split_and_compute_psi(self,
                                   value_col_actual: pd.Series,
                                   value_col_expected: pd.Series,
                                   bins_method: str,
                                   n_bins: int,
                                   zero_offset: float,
                                   return_psi_per_bin: bool = False,
                                   return_binned_col: bool = False,
                                   **kwargs) -> List[Any]:
        """
        Compute Population Stability Index using the input kind of binning.
        Supported binning methods: {spearman_corr, uniform}

        :param value_col_actual:        pd.Series, gotten actual values
        :param value_col_expected:      pd.Series, expected values
        :param bins_method:             str, kind of method which should be used to categorize
                                             actual and expected columns
        :param n_bins:                  int, number of bins used to make a column split in case `bins_method` = 'uniform',
                                             maximum number of bins used to make a 'spearman_corr' split
        :param y_true_expected:         pd.Series, optional column, which is used only
                                             when `bins_method` = 'uniform', and used as `y_true` parameter
        :param zero_offset:             float, usually small value used for adding to
                                             input values to avoid taking logarithm from zero
        :return:                        float, PSI value
        """
        feature_col = value_col_actual.name
        print(feature_col)
        if value_col_actual.empty or value_col_expected.empty:
            msg = "[Metrics] {} and/or {} empty. 0 will be returned".format(value_col_actual.name, value_col_expected.name)
            logging.warning(msg)
            warnings.warn(msg)
            return 0
        value_expected_bins, unique_bins = Splitter.split_by_quantiles(value_col_expected, n_quantiles=n_bins, retbins=True)
        act_split_values = Splitter().split_based_on_the_input_bins(
            value_col_actual=value_col_actual,
            value_expected_bins=unique_bins,
            col_name=feature_col, 
            epsilon=None
        )
        value_actual_bins = pd.DataFrame(
            {
                "bins": act_split_values[0],
                feature_col: value_col_actual
            }
        )
        exp_split_values = Splitter().split_based_on_the_input_bins(
            value_col_actual=value_col_expected,
            value_expected_bins=act_split_values[1],
            col_name=feature_col, epsilon=None
        )
        value_expected_bins = pd.DataFrame(
            {
                "bins": exp_split_values[0],
                feature_col: value_col_expected
            }
        )
        # compute distribution by bins
        value_actual = value_actual_bins["bins"].value_counts(normalize=True, sort=False).reset_index(name="freq_actual")
        value_expected = value_expected_bins["bins"].value_counts(normalize=True, sort=False).reset_index(name="freq_expected")

        value_actual.rename(columns={"index": "bins"}, inplace=True)
        value_expected.rename(columns={"index": "bins"}, inplace=True)

        if len(value_actual) != len(value_expected):
            msg = "[Metrics] `{}`: number of unique bins in actual values {} and in expected values {} can't be matched".format(
                value_col_actual.name, len(value_actual), len(value_expected)
            )
            logging.warning(msg)
            warnings.warn(msg)

        all_values = pd.merge(
            value_expected, value_actual, how="outer", on="bins"
        )

        psi_indexes = self.psi_total(
            value_col_actual=all_values["freq_actual"], value_col_expected=all_values["freq_expected"],
            zero_offset=zero_offset
        )
        return_vals = [None, None, dict()]
        if return_psi_per_bin:
            psi_per_bin = self.psi_index_per_value(
                value_col_actual=all_values["freq_actual"], value_col_expected=all_values["freq_expected"],
                zero_offset=zero_offset
            )
            psi_per_bin = pd.DataFrame(
                {
                    "bins": all_values.bins,
                    "psi": psi_per_bin
                }
            )
            psi_per_bin["left"] = psi_per_bin.bins.apply(lambda x: x.left)
            psi_per_bin.sort_values("left", inplace=True, ignore_index=True)
            del psi_per_bin["left"]
            return_vals[0], return_vals[1] = psi_indexes, psi_per_bin
        else:
            return_vals[0] = psi_indexes
        if return_binned_col:
            return_vals[-1] = {
                "selected_time_period": value_actual_bins["bins"],
                "benchmark": value_expected_bins["bins"],
            }
        return return_vals

    @staticmethod
    def _iv_computation(qq_df: pd.DataFrame, target_col: str, feature_col: str,
                        positive_inf_offset: float = 1.1,
                        negative_inf_offset: float = 1.1,
                        return_psi_per_bin: bool = False) -> (pd.DataFrame, Union[None, pd.DataFrame]):
        """
        Compute information value by the input feature's bins and target column:
        IV = sum_by_intervals[(%_of_negative - %_positive) * weight_of_evidence],
        where weight_of_evidence = ln(%_of_negative/%_positive)
        Note: for binary target only.

        :param qq_df:                  pd.Series, of target variable (note, only binary target should be used with
                                                    classes: positive = 1, negative = 0) and feature values
        :param target_col:             str, name of the target column
        :param feature_col:            str, name of the feature column
        :param positive_inf_offset:    float, weight of maximum value to fill +np.inf in information value
        :param negative_inf_offset:    float, weight of maximum value to fill -np.inf in information value
        :return:                       pd.DataFrame, with columns ['Feature', 'IV'], where the feature name and
                                                     its information value is stored
        """
        # split data into bins & count total number of observations and number of positive class per bin
        qq_df_stats = qq_df.groupby("bins")[target_col].sum().reset_index(name="n_positive")
        qq_df_stats["n"] = qq_df.groupby("bins")[target_col].count().values
        qq_df_stats["n_negative"] = qq_df_stats["n"] - qq_df_stats["n_positive"]

        # compute number of positive and negative class in the input data
        n_positive_df = (qq_df[target_col] == 1).sum()
        n_negative_df = (qq_df[target_col] == 0).sum()

        # weight of evidence
        qq_df_stats["woe"] = np.log(
            (qq_df_stats["n_negative"] / n_negative_df) / (qq_df_stats["n_positive"] / n_positive_df)
        )
        qq_df_stats["diff"] = (qq_df_stats["n_negative"] / n_negative_df - qq_df_stats["n_positive"] / n_positive_df)

        # woe smoothing
        try:
            max_woe = max(qq_df_stats['woe'].loc[qq_df_stats['woe'] != np.inf])
            max_woe = max_woe * positive_inf_offset
            min_woe = min(qq_df_stats['woe'].loc[qq_df_stats['woe'] != -np.inf])
            min_woe = min_woe * negative_inf_offset
        except:
            msg = "WOE smoothing can't be applied"
            warnings.warn(msg)
            max_woe = positive_inf_offset
            min_woe = negative_inf_offset

        # fill zero division
        qq_df_stats.woe.replace(np.inf, max_woe, inplace=True)
        qq_df_stats.woe.replace(-np.inf, min_woe, inplace=True)
        qq_df_stats.woe.replace(np.nan, 0, inplace=True)

        qq_df_stats["diff"].replace(np.inf, 0, inplace=True)
        qq_df_stats["diff"].replace(np.nan, 0, inplace=True)

        iv_per_bin = pd.DataFrame(
            {
                "IV_per_bin": qq_df_stats["diff"] * qq_df_stats["woe"],
                "bins": qq_df_stats.bins
            }
        )

        total_df = sum(iv_per_bin.IV_per_bin)

        iv_df = pd.DataFrame.from_dict(
            {"Feature": [feature_col], "IV": [total_df]}
        )
        if return_psi_per_bin:
            iv = (iv_df, pd.DataFrame(iv_per_bin))
        else:
            iv = (iv_df, None)
        return iv

    def iv_numeric(self,
                   df_actual: pd.DataFrame,
                   feature_col: str,
                   target_col: str,
                   n_bins: int,
                   positive_inf_offset: float,
                   negative_inf_offset: float,
                   return_psi_per_bin: bool = False,
                   preserved_split: bool = False) -> (pd.DataFrame, Union[None, pd.DataFrame]):
        """
        Compute information value for numeric feature: split numeric
        feature into specified number of quantiles and compute iv by bin using standard approach for categorical data.
        Note: for binary target only!

        :param df_actual:              pd.DataFrame, with target and corresponding feature
        :param target_col:             str, name of the target column
        :param n_bins:                 int, number of bins used for feature split
        :param feature_col:            str, name of the feature column
        :param positive_inf_offset:    float, weight of maximum value to fill +np.inf in information value
        :param negative_inf_offset:    float, weight of maximum value to fill -np.inf in information value
        :return:                       pd.DataFrame, with columns ['Feature', 'IV'], where the feature name and
                                                     its information value is stored
        """
        # split an input feature into quantiles where the target is not 2
        actual_values = df_actual.loc[df_actual[target_col] != 2, feature_col].reset_index(drop=True)
        if preserved_split:
            value_expected_bins = pd.DataFrame(self.__bins[feature_col], columns=["bins"])
            split_ = Splitter().split_based_on_the_input_bins(
                value_col_actual=actual_values,
                value_expected_bins=value_expected_bins,
                col_name=feature_col
            )
            qq_df = pd.DataFrame(
                {
                    "bins": split_[0],
                    feature_col: actual_values,
                }
            )
        else:
            qq_df = Splitter().split_by_quantiles(
                actual_values, n_bins
            )
        # concatenate input feature with the target
        qq_df = pd.concat(
            (qq_df.reset_index(drop=True),
             df_actual.loc[df_actual[target_col] != 2, target_col].reset_index(drop=True)), axis=1
        )
        iv = self._iv_computation(
            qq_df, target_col, feature_col, positive_inf_offset, negative_inf_offset, return_psi_per_bin
        )
        # replace too low negative values with 0, since that can be a value of approximate computations
        iv[0].loc[iv[0]["IV"] < -1e-6, "IV"] = 0
        return iv

    @classmethod
    def iv_categorical(cls,
                       df_actual: pd.DataFrame,
                       feature_col: str,
                       target_col: str,
                       positive_inf_offset: float,
                       negative_inf_offset: float,
                       return_psi_per_bin: bool = False,
                       preserved_split: bool = False) -> (pd.DataFrame, Union[None, pd.DataFrame]):
        """
        Compute information value for categorical feature using standard approach
        Note: for binary target only!

        :param df_actual:              pd.DataFrame, with target and corresponding feature
        :param target_col:             str, name of the target column
        :param feature_col:            str, name of the feature column
        :param positive_inf_offset:    float, weight of maximum value to fill +np.inf in information value
        :param negative_inf_offset:    float, weight of maximum value to fill -np.inf in information value
        :return:                       pd.DataFrame, with columns ['Feature', 'IV'], where the feature name and
                                                     its information value is stored
        """
        # transform the input data to the format needed for IV computation
        qq_df = df_actual.loc[df_actual[target_col] != 2, [feature_col, target_col]].copy()
        qq_df.rename(columns={feature_col: "bins"}, inplace=True)

        iv = cls._iv_computation(
            qq_df, target_col, feature_col, positive_inf_offset, negative_inf_offset, return_psi_per_bin
        )
        # replace too low negative values with 0, since that can be a value of approximate computations
        iv[0].loc[iv[0]["IV"] < -1e-6, "IV"] = 0
        return iv

    def psi_for_list_of_input_columns(self, list_of_columns: list,
                                      bins_method: str = "quantiles",
                                      n_bins: int = 10,
                                      zero_offset: float = 0.01,
                                      return_psi_per_bin: bool = False,
                                      preserve_split: bool = False,
                                      return_binned_col: bool = False) -> list:
        """
        Compute psi for each column and return in the format:
        [
            {
                "col_name": column_name,
                "psi": psi_value_in_numeric_format,
                "title": psi_value_inserted_into_title
            }, ...
       ]
        :param list_of_columns:     list, of columns in the format:
                                          [
                                            {
                                                "col_name": column_name,
                                                "values": {"benchmark": pd.Series, "selected_time_period": pd.Series},
                                                "type": "numeric"
                                            }, ...
                                          ];
        :param bins_method:         str, what kind of split to use for psi computation;
        :param n_bins:              int, number of bins used to split for psi computations;
        :param zero_offset:         float, values to change zeros in denominators while psi computations
        :return:                    list, with psi values using the described above format
        """
        subplot_metadata = []
        for col_info in list_of_columns:
            # try:
            plot_title = " ".join([elem for elem in col_info["col_name"].split("_") if elem != "_"])
            binned_values = dict()
            if col_info["type"] == "numeric":
                psi_values = self.make_split_and_compute_psi(
                    value_col_actual=col_info["values"]["selected_time_period"],
                    value_col_expected=col_info["values"]["benchmark"],
                    bins_method=bins_method, n_bins=n_bins,
                    zero_offset=zero_offset,
                    return_psi_per_bin=True,
                    return_binned_col=return_binned_col
                )
                psi_value = psi_values[0]
                binned_values = psi_values[-1]
                if preserve_split:
                    self.__bins[col_info["col_name"]] = psi_values[1]["bins"]
                color = "#2EE86A" * (psi_value < 0.1) + "orange" * (
                            (psi_value >= 0.1) & (psi_value <= 0.25)) + "red" * (psi_value > 0.25)
            elif col_info["type"] == "categorical":
                psi_value = self.psi_total(
                    value_col_actual=col_info["values"]["selected_time_period"].value_counts(normalize=True),
                    value_col_expected=col_info["values"]["benchmark"].value_counts(normalize=True),
                    zero_offset=zero_offset
                )

                slctd = col_info["values"]["selected_time_period"].value_counts(normalize=True).reset_index(
                    name="slctd"
                ).rename(columns={"index": "bins"})
                bnchmrk = col_info["values"]["benchmark"].value_counts(normalize=True).reset_index(
                    name="bnchmrk").rename(columns={"index": "bins"})
                overall = pd.merge(slctd, bnchmrk, how="outer", on="bins")
                overall.slctd.fillna(0, inplace=True)
                overall.bnchmrk.fillna(0, inplace=True)

                if preserve_split:
                    self.__bins[col_info["col_name"]] = overall["bins"]

                psi_values = [None, self.psi_index_per_value(
                    value_col_actual=overall.slctd,
                    value_col_expected=overall.bnchmrk,
                    zero_offset=zero_offset
                )]
                psi_values[1] = pd.DataFrame(psi_values[1])
                psi_values[1]["bins"] = overall["bins"]
                color = "#2EE86A" * (psi_value < 0.1) + "orange" * (
                            (psi_value >= 0.1) & (psi_value <= 0.25)) + "red" * (psi_value > 0.25)

            plot_title += "<br><span style='color:{color}';>(PSI={psi})</span>".format(
                psi=round(psi_value, 3),
                color=color
            )
            d = {
                "col_name": col_info["col_name"],
                "psi": psi_value,
                "title": plot_title
            }
            if return_psi_per_bin:
                d.update({"psi_per_bin": psi_values[1]})
            if return_binned_col:
                d.update({"binned_values": binned_values})
            subplot_metadata += [d]
            # except:
            #     warnings.warn(f"Problem with: {col_info['col_name']}")

        return subplot_metadata

    def iv_for_list_of_input_columns(self,
                                     df: pd.DataFrame,
                                     list_of_columns: list,
                                     target_col_name: str,
                                     n_bins: int,
                                     positive_inf_offset: float,
                                     negative_inf_offset: float,
                                     return_psi_per_bin: bool = False,
                                     preserved_split: bool = False
                                     ) -> pd.DataFrame:
        """
        Compute Information Value for the each input column and return pandas data frame with columns: [Feature, IV]

        :param df:                               pd.Series, with features and target;
        :param list_of_columns:                  list, of columns in the format:
                                                      [
                                                        {
                                                            "col_name": column_name,
                                                            "type": "numeric"
                                                        }, ...
                                                      ];
        :param target_col_name:                  str, name of target column;
        :param n_bins:                           int, number of bins used to split for psi computations;
        :param positive_inf_offset:              float,
        :param negative_inf_offset:              float,
        :return:                                 pd.DataFrame, with psi values using the described above format
        """
        iv_data = pd.DataFrame()
        for col_info in list_of_columns:
            params = dict(
                df_actual=df, feature_col=col_info["col_name"], target_col=target_col_name, n_bins=n_bins,
                positive_inf_offset=positive_inf_offset, negative_inf_offset=negative_inf_offset,
                return_psi_per_bin=return_psi_per_bin, preserved_split=preserved_split
            )

            if col_info["type"] == "categorical":
                del params["n_bins"]
            iv_df, iv_per_bin = getattr(self, "iv_" + col_info["type"])(**params)
            if return_psi_per_bin:
                iv_per_bin = pd.DataFrame(
                    {
                        "IV_per_bin": [iv_per_bin]
                    }
                )
                iv_df = pd.concat(
                    (
                        iv_df,
                        iv_per_bin
                    ),
                    axis=1
                )
            iv_data = pd.concat(
                [
                    iv_data,
                    iv_df
                ],
                axis=0
            )
        return iv_data

    @classmethod
    def gini_per_time_interval(cls, df: pd.DataFrame,
                               params_cols_match: Dict,
                               n_cut_off: int = 2000):
        """
        # temp = pd.DataFrame(
        #     {
        #         "_date": pd.to_datetime(["31/03/2021", "30/04/2021", "31/05/2021", "30/06/2021", "31/07/2021", "31/08/2021", "30/09/2021", "31/10/2021", "30/11/2021", "31/12/2021", "31/01/2022"]).date,
        #         "n": [1, 190, 648, 435, 336, 1535, 3176, 1288, 2153, 147, 18]
        #     }
        # )
        """
        df["_date"] = pd.to_datetime(df[params_cols_match["date"]]).dt.date + pd.offsets.MonthEnd(0)

        df_count_per_month = df.groupby(["_date"])[params_cols_match["date"]].count().reset_index(name="n")
        df_count_per_month.sort_values(["_date"], inplace=True, ignore_index=True)

        cum_sum = 0
        dates_to_compute = []
        gini_list = []
        i = 0
        while i <= df_count_per_month.shape[0]:
            if i != df_count_per_month.shape[0]:
                r = df_count_per_month.iloc[i, :]
        # for i, r in df_count_per_month.iterrows():
            if cum_sum < n_cut_off:
                cum_sum += r["n"]
                dates_to_compute += [r["_date"]]
            elif cum_sum >= n_cut_off:
                # gini by intervals
                min_date = min(dates_to_compute)
                max_date = max(dates_to_compute)

                metrics = cls(
                    metrics=["gini"],
                    data=df[(df._date >= min_date) & (df._date <= max_date)],
                    params_cols_match=params_cols_match,
                )
                metrics_values = metrics.compute()
                gini_list += [((min_date, max_date), metrics_values["other"]["gini"])]
                if i != df_count_per_month.shape[0]:
                    cum_sum = r["n"]
                    dates_to_compute = [r["_date"]]
            i += 1
        del df["_date"]
        return gini_list

    @classmethod
    def metrics_per_time_interval(cls,
                                  data: pd.DataFrame,
                                  metrics_list: List[str],
                                  target_name: str,
                                  algo_name: str,
                                  date_col: str,
                                  date_type: str
                                  ) -> pd.DataFrame:
        if date_type == "m":
            date_col_vals = data[date_col] + MonthEnd(0)
        elif date_type == "Q":
            date_col_vals = pd.to_datetime(data[date_col]).dt.to_period('Q')

        date_to_iterate = date_col_vals.unique()
        df_return = pd.DataFrame()
        for d in date_to_iterate:
            df_temp = data[date_col_vals == d].reset_index(drop=True)
            nulls_mask = df_temp[algo_name].isnull()
            if not df_temp[~nulls_mask].empty:
                metrics_benchmark = cls(
                    metrics=metrics_list,
                    data=df_temp[~nulls_mask].reset_index(drop=True),
                    params_cols_match={"y_true": target_name, "y_prob": algo_name},
                )
                del nulls_mask

                metrics_values_benchmark = metrics_benchmark.compute()
                if metrics_values_benchmark.get("threshold_dependent") is not None:
                    metrics_values_benchmark["other"].update(metrics_values_benchmark.get("threshold_dependent"))
                df_return_temp = pd.DataFrame(metrics_values_benchmark)[["other"]]
                df_return_temp["Date"] = d

                df_return = pd.concat((df_return_temp, df_return))
        df_return.rename(columns={"other": "values"}, inplace=True)
        return df_return

    fraud_detection_rate = recall
    psi = psi_index_per_value
