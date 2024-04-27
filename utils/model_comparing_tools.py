import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Iterable
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from core.tools.metrics.metrics_by_quantile import MetricsByQuantile


# def scores_distributions(df: pd.DataFrame,
#                          model_name: str,
#                          target_col_name: str = "y_true",
#                          titles: Iterable = ("Distribution of probabilities (train data)",
#                                              "Distribution of probabilities (test data)")):
#     # Scores' distributions
#     # ----------------------------------------------
#     f = plt.figure(figsize=(20, 5), dpi=250)
#     _ = sns.histplot(x=model_name, data=df[df.data_part == "train"], hue=target_col_name, common_norm=False,
#                      stat="density", color="orange")
#     _ = plt.title(titles[0], fontdict={"size": 20})

#     f = plt.figure(figsize=(20, 5), dpi=250)
#     _ = sns.histplot(x=model_name, data=df[df.data_part == "test"], hue=target_col_name, common_norm=False,
#                      stat="density", color="blue")
#     _ = plt.title(titles[1], fontdict={"size": 20})

#     f = plt.figure(figsize=(20, 5), dpi=250)
#     _ = sns.histplot(x=model_name, data=df[df.data_part == "cv_test"], hue=target_col_name, common_norm=False,
#                      stat="density", color="orange")
#     _ = plt.title(titles[2], fontdict={"size": 20})

#     plt.show()


# def compare_scores_distributions(df: pd.DataFrame,
#                                  model_name: str,
#                                  bw_adjust=2,
#                                  data_parts: list = [("train", "test"), ("cv_test", "test")],
#                                  palette: dict = {"train": "orange", "test": "blue", "cv_test": "orange"},
#                                  titles: Iterable = ("Distribution of probabilities (train, test data)",
#                                                      "Distribution of probabilities (cv, test data)"),
#                                  n_bins: int = None,):
#     # Scores' distributions
#     # ----------------------------------------------
#     for i, data_pair in enumerate(data_parts):
#         f = plt.figure(figsize=(20, 5), dpi=250)
#         _ = sns.kdeplot(
#             x=model_name,
#             data=df[(df.data_part == data_pair[0]) | (df.data_part == data_pair[1])].reset_index(drop=True),
#             hue="data_part",
#             palette=palette,
#             common_norm=False,
#             bw_adjust=bw_adjust
#         )
#         # _ = sns.histplot(
#         #     x=model_name,
#         #     data=df[(df.data_part == data_parts[0][0]) | (df.data_part == data_parts[0][1])].reset_index(drop=True),
#         #     hue="data_part",
#         #     palette=palette,
#         #     common_norm=False, kde=True,
#         #     stat="density", bins=n_bins
#         # )
#         _ = plt.title(titles[i], fontdict={"size": 20})

#     # f = plt.figure(figsize=(20, 5), dpi=250)
#     # _ = sns.histplot(
#     #     x=model_name,
#     #     data=df[(df.data_part == data_parts[1][0]) | (df.data_part == data_parts[1][1])].reset_index(drop=True),
#     #     hue="data_part",
#     #     palette=palette,
#     #     common_norm=False, kde=True,
#     #     stat="density", bins=n_bins
#     # )
#     # _ = plt.title(titles[1], fontdict={"size": 20})
#     plt.show()


# def prec_recall_auc(df_dict: dict):
#     prec_recal_df = pd.DataFrame()
#     for model_alias, data in df_dict.items():
#         pr_auc_value = auc(data["recall"], data["precision"])

#         prec_recal_df = pd.concat(
#             (
#                 prec_recal_df,
#                 pd.DataFrame.from_dict(
#                     {
#                         "Model": model_alias,
#                         "PR_AUC": pr_auc_value
#                     },
#                     orient="index"
#                 ).T
#             )
#         )
#     return prec_recal_df


# def roc_auc_by_dict(df_dict: dict):
#     fpr_tpr_df = pd.DataFrame()
#     for model_alias, data in df_dict.items():
#         roc_auc_value = auc(data["fpr"], data["tpr"])

#         fpr_tpr_df = pd.concat(
#             (
#                 fpr_tpr_df,
#                 pd.DataFrame.from_dict(
#                     {
#                         "Model": model_alias,
#                         "ROC_AUC": roc_auc_value
#                     },
#                     orient="index"
#                 ).T
#             )
#         )
#     return fpr_tpr_df


# def train_test_cv_roc_auc_compare(df: pd.DataFrame,
#                                   model_name: str,
#                                   target_col_name: str,
#                                   data_parts: tuple = ("cv_test", "test"),
#                                   titles: Iterable = ("Receiver Operating Characteristic. CV & OOT"),
#                                   palette: dict = {"cv_test": "orange", "test": "blue"},
#                                   n_std: int = 2,
#                                   ci: bool = True,
#                                   on_cv: bool = True,
#                                   n: int =18000):
#     # ROC AUC curve: comparing train & test performance
#     # ---------------------------------------------------
#     # computations
#     fpr_test, tpr_test, thresholds_test = roc_curve(df[df.data_part == data_parts[1]][target_col_name],
#                                                     df[df.data_part == data_parts[1]][model_name])
#     if on_cv:
#         tpr_cv_test_list = []
#         auc_cv_test_list = []
#         fpr_cv_test_list = []

#         for i in range(df[df.data_part == data_parts[0]].cv_iter.astype(int).min(),
#                        df[df.data_part == data_parts[0]].cv_iter.astype(int).max()+1):
#             fpr_cv_test, tpr_cv_test, thresholds_cv_test = roc_curve(
#                 df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][target_col_name],
#                 df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][model_name]
#             )
#             # auc_cv = roc_auc_score(
#             #     df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][target_col_name],
#             #     df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][model_name]
#             # )
#             np.random.seed(42)
#             indx = np.sort(np.random.choice(np.arange(0, len(thresholds_cv_test)), n, replace=False))
#             fpr_cv_test = fpr_cv_test[indx]
#             tpr_cv_test = tpr_cv_test[indx]

#             tpr_cv_test_list.append(tpr_cv_test)
#             fpr_cv_test_list.append(fpr_cv_test)
#             auc_cv_test_list.append(fpr_cv_test)

#         tpr_cv_test_mean = np.mean(tpr_cv_test_list, axis=0)
#         fpr_cv_test_mean = np.mean(fpr_cv_test_list, axis=0)

#         # tpr_cv_test_mean[-1] = 1.0

#         tpr_cv_test_std = np.std(tpr_cv_test_list, axis=0)

#         tprs_upper = np.minimum(tpr_cv_test_mean + n_std * tpr_cv_test_std, 1)
#         tprs_lower = np.maximum(tpr_cv_test_mean - n_std * tpr_cv_test_std, 0)

#     # plot train/test
#     if not on_cv:
#         fpr, tpr, thresholds = roc_curve(df[df.data_part == data_parts[0]][target_col_name],
#                                          df[df.data_part == data_parts[0]][model_name])
#         f = plt.subplots(figsize=(20, 5))
#         _ = plt.plot(fpr, tpr, label=data_parts[0], color=palette[data_parts[0]])
#         _ = plt.plot(fpr_test, tpr_test, label=data_parts[1], color=palette[data_parts[1]])
#         _ = plt.title(titles[0], fontdict={"size": 20})
#         _ = plt.xlabel("False Positive Rate", fontdict={"size": 15})
#         _ = plt.ylabel("True Positive Rate", fontdict={"size": 15})
#         _ = plt.legend()
#         plt.show()

#     if on_cv:
#         # plot CV/test
#         f = plt.subplots(figsize=(20, 5))
#         _ = plt.plot(fpr_cv_test_mean, tpr_cv_test_mean, label="Mean CV", color=palette[data_parts[0]])
#         _ = plt.plot(fpr_test, tpr_test, label=data_parts[1], color=palette[data_parts[1]])

#     if ci:
#         _ = plt.fill_between(
#             fpr_cv_test_mean,
#             tprs_lower,
#             tprs_upper,
#             color="grey",
#             alpha=0.8,
#             label=f"$\pm$ {n_std} std. dev.",
#         )
#         _ = plt.title(titles[0], fontdict={"size": 20})
#         _ = plt.xlabel("False Positive Rate", fontdict={"size": 15})
#         _ = plt.ylabel("True Positive Rate", fontdict={"size": 15})
#         _ = plt.legend()
#         plt.show()


# def train_test_cv_pr_recall_curve_compare(df: pd.DataFrame,
#                                           model_name: str,
#                                           target_col_name: str,
#                                           model_alias: str = None,
#                                           data_parts: tuple = ("train", "test"),
#                                           palette: dict = {"cv_test": "orange", "test": "blue"},
#                                           titles: Iterable = ("Receiver Operating Characteristic. Train & OOT",),
#                                           n_std: int = 2,
#                                           ci: bool = False,
#                                           on_cv: bool = False,
#                                           n: int = 200000):
#     # Precision-Recall Curve: comparing train & test performance
#     # ----------------------------------------------------------
#     # computations

#     prec_test, recall_test, thresholds_test = precision_recall_curve(
#         df[df.data_part == data_parts[1]][target_col_name], df[df.data_part == data_parts[1]][model_name]
#     )
#     if not on_cv:
#         # plot
#         prec, recall, _ = precision_recall_curve(
#             df[df.data_part == data_parts[0]][target_col_name], df[df.data_part == data_parts[0]][model_name]
#         )
#         f = plt.figure(figsize=(20, 5), dpi=60)
#         g = sns.lineplot(x=recall, y=prec, label=data_parts[0], color=palette[data_parts[0]], ci=None)
#         g = sns.lineplot(x=recall_test, y=prec_test, label=data_parts[1], color=palette[data_parts[1]], ci=None)
#         _ = plt.title(titles[0], fontdict={"size": 20})
#         _ = plt.xlabel("Recall")
#         _ = plt.ylabel("Precision")
#         plt.show()

#         pr_auc_train_value = auc(recall, prec)
#         pr_auc_test_value = auc(recall_test, prec_test)

#         print(f"{data_parts[0]}: ", pr_auc_train_value)
#         print(f"{data_parts[1]}: ", pr_auc_test_value)

#     if on_cv:
#         prec_cv_test_list = []
#         recall_cv_test_list = []

#         auc_cv_test_list = []
#         for i in range(df[df.data_part == data_parts[0]].cv_iter.astype(int).min(),
#                        df[df.data_part == data_parts[0]].cv_iter.astype(int).max() + 1):
#             prec_cv_test, recall_cv_test, thresholds_cv_test = precision_recall_curve(
#                 df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][target_col_name],
#                 df[(df.data_part == data_parts[0]) & (df.cv_iter == i)][model_name]
#             )
#             auc_cv = auc(recall_cv_test, prec_cv_test)

#             np.random.seed(42)
#             indx = np.sort(np.random.choice(np.arange(0, len(thresholds_cv_test)), n, replace=False))
#             prec_cv_test = prec_cv_test[indx]
#             recall_cv_test = recall_cv_test[indx]

#             prec_cv_test_list.append(prec_cv_test)
#             recall_cv_test_list.append(recall_cv_test)
#             auc_cv_test_list += [auc_cv]

#         auc_cv_test_mean = np.mean(auc_cv_test_list, axis=0)
#         prec_cv_test_mean = np.mean(prec_cv_test_list, axis=0)
#         prec_cv_test_std = np.std(prec_cv_test_list, axis=0)

#         recall_cv_test_mean = np.mean(recall_cv_test_list, axis=0)

#         prec_upper = np.minimum(prec_cv_test_mean + n_std * prec_cv_test_std, 1)
#         prec_lower = np.maximum(prec_cv_test_mean - n_std * prec_cv_test_std, 0)

#         # plot
#         f = plt.figure(figsize=(20, 5), dpi=60)
#         g = sns.lineplot(x=recall_cv_test_mean, y=prec_cv_test_mean, label=data_parts[0], color=palette[data_parts[0]], ci=None)
#         g = sns.lineplot(x=recall_test, y=prec_test, label=data_parts[1], color=palette[data_parts[1]], ci=None)

#         print("Mean CV: ", auc_cv_test_mean)
#         print("OOT: ",  auc(recall_test, prec_test))

#     if ci:
#         _ = plt.fill_between(
#             recall_cv_test_mean,
#             prec_lower,
#             prec_upper,
#             color="grey",
#             alpha=0.8,
#             label=f"$\pm$ {n_std} std. dev.",
#         )

#         _ = plt.title(titles[0], fontdict={"size": 20})
#         _ = plt.xlabel("Recall")
#         _ = plt.ylabel("Precision")
#         plt.show()


# def plot_precision_recall_curves(df_dict: pd.DataFrame,
#                                  pred_col_name: str = "y_pred",
#                                  target_col_name: str = "TargetOutcome",
#                                  title: str = "Precision-Recall Curve",
#                                  palette: str = "tab10",
#                                  legend_params: dict = dict(),
#                                  data_part_name: str = "test"):
#     # Precision-Recall Curve: comparing train & test performance
#     # ----------------------------------------------------------
#     f = plt.figure(figsize=(20, 5), dpi=60)
#     prec_recall_curve_dict = dict()

#     # computations
#     for model_alias, data in tqdm(df_dict.items()):
#         prec_val, recall_val, thresholds_val = precision_recall_curve(
#             data[data.data_part == data_part_name][target_col_name],
#             data[data.data_part == data_part_name][pred_col_name]
#         )
#         prec_recall_curve_dict[model_alias] = {
#             "precision": copy.deepcopy(prec_val),
#             "recall": copy.deepcopy(recall_val),
#             "thresholds": copy.deepcopy(thresholds_val)
#         }
#         # plot
#         g = sns.lineplot(x=recall_val, y=prec_val, label=model_alias, palette=palette, ci=None)

#     _ = plt.title(title, fontdict={"size": 20})
#     _ = plt.xlabel("Recall")
#     _ = plt.ylabel("Precision")
#     plt.legend(**legend_params)
#     plt.show()

#     return prec_recall_curve_dict


# def plot_roc_curves(df_dict: pd.DataFrame,
#                      pred_col_name: str = "y_pred",
#                      target_col_name: str = "TargetOutcome",
#                      title: str = "Precision-Recall Curve",
#                      palette: str = "tab10",
#                      legend_params: dict = dict(),
#                      data_part_name: str = "test"):
#     # ROC Curves on OOT for different models
#     # ----------------------------------------------------------
#     f = plt.figure(figsize=(20, 5), dpi=60)
#     fpr_tpr_curve_dict = dict()

#     # computations
#     for model_alias, data in tqdm(df_dict.items()):
#         fpr, tpr, thresholds = roc_curve(
#             data[data.data_part == data_part_name][target_col_name],
#             data[data.data_part == data_part_name][pred_col_name]
#         )
#         fpr_tpr_curve_dict[model_alias] = {
#             "fpr": copy.deepcopy(fpr),
#             "tpr": copy.deepcopy(tpr),
#             "thresholds": copy.deepcopy(thresholds)
#         }
#         # plot
#         g = sns.lineplot(x=fpr, y=tpr, label=model_alias, palette=palette, ci=None)

#     _ = plt.title(title, fontdict={"size": 20})
#     _ = plt.ylabel("True Positive Ratio")
#     _ = plt.xlabel("False Positive Ratio")
#     plt.legend(**legend_params)
#     plt.show()

#     return fpr_tpr_curve_dict


# def plot_recall_at_precision_bar(data: pd.DataFrame,
#                                  title: str, hue_col_name: str = None, annotations: bool = True,
#                                  legend_params: dict = None, palette: str = "tab10"):
#     f = plt.figure(figsize=(10, 4), dpi=250)

#     __ = sns.barplot(
#         x="Expected Precision", y="Recall", hue=hue_col_name, data=data,
#         color="#D3D3D3", palette=palette
#     )
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))

#     locs, labels = plt.xticks()

#     _ = sns.lineplot(
#         x=list(locs) * data[hue_col_name].nunique(), y="Recall", hue=hue_col_name, data=data,
#         color="#D3D3D3", palette=palette, ci=None
#     )

#     _ = plt.title(title, fontdict={"size": 15})
#     _ = plt.xlabel("Expected Precision", fontdict={"size": 10})
#     _ = plt.ylabel("Recall", fontdict={"size": 10})

#     if annotations:
#         for i, l in enumerate(locs):
#             _ = plt.annotate(
#                 str(round(data["Recall"].values[i], 2)) + "%",
#                 xy=(l, data["Recall"].values[i]),
#                 ha='center', va='bottom',
#                 fontsize=8
#             )
#     if legend_params:
#         plt.legend(by_label.values(), by_label.keys(), **legend_params)


# def plot_num_of_observations(df: pd.DataFrame, title: str, y_lim: float, y_text: float,
#                              train_start: pd.Timestamp, train_end: pd.Timestamp,
#                              test_start: pd.Timestamp, test_end: pd.Timestamp):
#     df_dates = df.groupby(["ModelScoreDate_MonthEnd"]).ACCNO.count().reset_index(name="n_payments").copy()
#     df_dates["ModelScoreDate_MonthEnd"] = pd.to_datetime(df_dates["ModelScoreDate_MonthEnd"]).dt.date

#     df_dates = pd.merge(
#         df_dates,
#         df.groupby(["ModelScoreDate_MonthEnd"]).TargetOutcome.sum().reset_index(name="n_pos").copy(),
#         how="inner", on="ModelScoreDate_MonthEnd"
#     )

#     df_dates["data_part"] = None
#     df_dates.loc[(df_dates.ModelScoreDate_MonthEnd >= pd.to_datetime(train_start)) & (
#                 df_dates.ModelScoreDate_MonthEnd < pd.to_datetime(train_end)), "data_part"] = "train"
#     df_dates.loc[(df_dates.ModelScoreDate_MonthEnd >= pd.to_datetime(test_start)) & (
#                 df_dates.ModelScoreDate_MonthEnd < pd.to_datetime(test_end)), "data_part"] = "test"

#     df_dates["color"] = None
#     df_dates.loc[df_dates.data_part == "train", "color"] = "#D3D3D3"
#     df_dates.loc[df_dates.data_part == "test", "color"] = "#696969"

#     f = plt.figure(figsize=(20, 5), dpi=60)

#     g = sns.barplot(x="ModelScoreDate_MonthEnd", y="n_payments", data=df_dates, palette=df_dates.color.values)
#     _ = plt.title(title + " # observations", fontdict=dict(size=20))
#     _ = plt.xticks(rotation=45, labels=df_dates.ModelScoreDate_MonthEnd, ticks=list(range(len(df_dates))))
#     _ = plt.xlabel("Month")
#     _ = plt.vlines(11.5, -10, y_lim, colors="black", linestyles="dashed")

#     _ = plt.text(5, y_text, "Train", fontdict=dict(size=15))
#     _ = plt.text(12.6, y_text, "Test", fontdict=dict(size=15))

#     for index, row in df_dates.iterrows():
#         g.text(index, row.n_payments + 5000, f"{row.n_payments}", color='black', ha="center", fontdict=dict(size=15))

#     _ = sns.lineplot(x=list(range(len(df_dates))), y="n_payments", data=df_dates, color="grey", ci=None)

#     plt.show()
#     return df_dates


# def plot_num_of_observations_by_positive_class(df: pd.DataFrame, title: str, y_lim: float, y_text: float):
#     f = plt.figure(figsize=(20, 5), dpi=60)

#     g = sns.barplot(x="ModelScoreDate_MonthEnd", y="n_pos", data=df, palette=df.color.values)
#     _ = sns.lineplot(x=list(range(len(df))), y="n_pos", data=df, color="grey", ci=None)
#     _ = plt.title(title + " # payments from positive class by month", fontdict=dict(size=20))
#     _ = plt.xticks(rotation=45, labels=df.ModelScoreDate_MonthEnd, ticks=list(range(len(df))))
#     _ = plt.ylim(0, y_lim)
#     _ = plt.xlabel("Month")

#     _ = plt.vlines(11.5, -10, y_text + 0.10 * y_text, colors="black", linestyles="dashed")
#     _ = plt.text(4.5, y_text, "Train", fontdict=dict(size=15))
#     _ = plt.text(13, y_text, "Test", fontdict=dict(size=15))

#     for index, row in df.iterrows():
#         g.text(index, row.n_pos+10, f"{row.n_pos}", color='black', ha="center", fontdict=dict(size=15))


# def compute_recall_at_precision(df_deciles: pd.DataFrame, precision_val_list: list, model_alias: str):
#     df_recall_at_precision = pd.DataFrame()
#     for prec_value in precision_val_list:
#         argmin_indx = (df_deciles["Cumulative Precision"] - prec_value).abs().argmin()
#         df_prec_recall_temp = pd.DataFrame.from_dict(
#             {
#                 "Scores": df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index("Scores")],
#                 "Precision": df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index("Cumulative Precision")],
#                 "Recall": df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index("Cumulative Recall")],
#                 "Cumulative % of total population": df_deciles.iloc[argmin_indx, df_deciles.columns.to_list().index("Cumulative % of total population")],
#                 "Model": model_alias,
#                 "Expected Precision": prec_value
#             }, orient="index"
#         )
#         df_prec_recall_temp = df_prec_recall_temp.T
#         df_recall_at_precision = pd.concat((df_recall_at_precision, df_prec_recall_temp))
#         df_recall_at_precision.reset_index(drop=True, inplace=True)
#     return df_recall_at_precision


# def compute_recall_at_precision_for_dict(data_dict: dict,
#                                          precision_val_list: list,
#                                          n_quantiles: int = 10000,
#                                          target_col_name: str = "TargetOutcome"):
#     recall_at_precision_dict = dict()
#     i = 0
#     for alias, data in data_dict.items():
#         data_train = data[(data.data_part == "train")]
#         data_test = data[(data.data_part == "test")]

#         data_trdeciles, data_tdeciles = MetricsByQuantile().quantiles_analytics_by_benchmark_and_selected_period(
#             y_benchmark={
#                 "y_true": data_train[target_col_name].reset_index(drop=True),
#                 "y_pred": data_train["y_pred"].reset_index(drop=True)
#             },
#             y_slctd_period={
#                 "y_true": data_test[target_col_name].reset_index(drop=True),
#                 "y_pred": data_test["y_pred"].reset_index(drop=True)
#             },
#             list_of_metrics=["cum_precision_by_quantile", "cum_recall_by_quantile"],
#             n_quantiles=n_quantiles,
#         )

#         recall_at_precision_dict[alias] = compute_recall_at_precision(
#             df_deciles=data_tdeciles.reset_index(drop=True), precision_val_list=precision_val_list, model_alias=alias
#         )
#     return recall_at_precision_dict


# def compute_recall_at_precision_for_cv(data: pd.DataFrame,
#                                        precision_val_list: list,
#                                        n_quantiles: int = 10000,
#                                        target_col_name: str = "TargetOutcome"):
#     recall_at_precision_df = pd.DataFrame()
#     cv_iter = data["cv_iter"].astype(int).unique()

#     for i in cv_iter:
#         sample = data[data["cv_iter"] == i]

#         data_trdeciles, data_tdeciles = MetricsByQuantile().quantiles_analytics_by_benchmark_and_selected_period(
#             y_benchmark={
#                 "y_true": sample[target_col_name].reset_index(drop=True),
#                 "y_pred": sample["y_pred"].reset_index(drop=True)
#             },
#             y_slctd_period={
#                 "y_true": sample[target_col_name].reset_index(drop=True),
#                 "y_pred": sample["y_pred"].reset_index(drop=True)
#             },
#             list_of_metrics=["cum_precision_by_quantile", "cum_recall_by_quantile"],
#             n_quantiles=n_quantiles,
#         )
#         recall_at_precision_df = pd.concat(
#             (
#                 recall_at_precision_df,
#                 compute_recall_at_precision(
#                     df_deciles=data_tdeciles.reset_index(drop=True),
#                     precision_val_list=precision_val_list,
#                     model_alias=""
#                 ),
#             )
#         )
#     mean_df = recall_at_precision_df.groupby("Expected Precision").Recall.mean().reset_index(name="Mean Recall")
#     mean_df["std"] = recall_at_precision_df.groupby("Expected Precision").Recall.std().values
#     mean_df["Mean Real Precision"] = recall_at_precision_df.groupby("Expected Precision").Precision.mean().values
#     return mean_df


# def compute_recall_at_precision_for_oots(data: dict,
#                                          precision_val_list: list,
#                                          alias: str,
#                                          n_quantiles: int = 10000,
#                                          target_col_name: str = "TargetOutcome"):
#     data_trdeciles, data_tdeciles = MetricsByQuantile().quantiles_analytics_by_benchmark_and_selected_period(
#         y_benchmark={
#             "y_true": data[target_col_name].reset_index(drop=True),
#             "y_pred": data["y_pred"].reset_index(drop=True)
#         },
#         y_slctd_period={
#             "y_true": data[target_col_name].reset_index(drop=True),
#             "y_pred": data["y_pred"].reset_index(drop=True)
#         },
#         list_of_metrics=["cum_precision_by_quantile", "cum_recall_by_quantile"],
#         n_quantiles=n_quantiles,
#     )

#     recall_at_precision_df = compute_recall_at_precision(
#         df_deciles=data_tdeciles.reset_index(drop=True), precision_val_list=precision_val_list, model_alias=alias
#     )
#     return recall_at_precision_df


# def train_test_metrics_compare(df: pd.DataFrame,
#                                model_name: str,
#                                target_col_name: str,
#                                model_alias: str,
#                                titles: list = [
#                                    "Receiver Operating Characteristic", "Precision-Recall Curve"
#                                ], 
#                                color_dict: dict = {"train": "orange", "test": "blue"}):
#     # ROC AUC curve: comparing train & test performance
#     # ---------------------------------------------------
#     # computations
#     prec_recall_curve_dict = dict()
#     roc_curve_dict = dict()
    
#     # plot
#     f = plt.subplots(figsize=(20, 5))
#     for data_part_value in df.data_part.unique():
#         fpr, tpr, thresholds = roc_curve(df[df.data_part == data_part_value][target_col_name],
#                                          df[df.data_part == data_part_value][model_name])

#         roc_curve_dict[model_alias] = {
#             f"fpr_{data_part_value}": copy.deepcopy(fpr),
#             f"tpr_{data_part_value}": copy.deepcopy(tpr),
#             f"thresholds_{data_part_value}": copy.deepcopy(thresholds),
#         }

#         _ = plt.plot(fpr, tpr, label=data_part_value)
#     _ = plt.title(titles[0], fontdict={"size": 20})
#     _ = plt.xlabel("False Positive Rate", fontdict={"size": 15})
#     _ = plt.ylabel("True Positive Rate", fontdict={"size": 15})
#     _ = plt.legend()
#     plt.show()

#     # Precision-Recall Curve: comparing train & test performance
#     # ----------------------------------------------------------
#     # computations
#     # plot
#     f = plt.figure(figsize=(20, 5), dpi=60)
#     for data_part_value in df.data_part.unique():
#         prec, recall, threshold = precision_recall_curve(df[df.data_part == data_part_value][target_col_name],
#                                                  df[df.data_part == data_part_value][model_name])
#         prec_recall_curve_dict[model_alias] = {
#             f"precision_{data_part_value}": copy.deepcopy(prec),
#             f"recall_{data_part_value}": copy.deepcopy(recall),
#             f"thresholds_{data_part_value}": copy.deepcopy(threshold)
#         }
#         roc_auc_value = roc_auc_score(
#             df[df.data_part == data_part_value][target_col_name],
#             df[df.data_part == data_part_value][model_name]
#         )
#         pr_auc_value = auc(recall, prec)

#         print(f"{data_part_value}: ROC AUC: {roc_auc_value}; PR AUC: {pr_auc_value}")

#         roc_curve_dict[model_alias][f"roc_auc_{data_part_value}_value"] = roc_auc_value
#         prec_recall_curve_dict[model_alias][f"pr_auc_{data_part_value}_value"] = pr_auc_value
        
#         # plot
#         g = sns.lineplot(
#             x=recall, y=prec, label=data_part_value, 
#             color=color_dict[data_part_value], 
#             ci=None
#         )

#     _ = plt.title(titles[1], fontdict={"size": 20})
#     _ = plt.xlabel("Recall")
#     _ = plt.ylabel("Precision")
#     plt.show()

        

#     return roc_curve_dict, prec_recall_curve_dict


# def compare_metric_oot1_oot2(oot1: pd.DataFrame, oot2: pd.DataFrame, metric_name: str, metric_format: str = None):
#     oot = pd.concat((oot1.reset_index(drop=True), oot2.reset_index(drop=True)), axis=1, keys=["OOT 1", "OOT 2"])

#     oot[("Overall", "% Change")] = (oot[("OOT 2", metric_name)] - oot[("OOT 1", metric_name)]) / oot[
#         ("OOT 1", metric_name)]
#     oot[("Overall", "Model")] = oot[("OOT 1", "Model")]

#     oot = pd.concat(
#         (
#             pd.concat((oot[("Overall", "Model")], oot[("Overall", "% Change")]), axis=1),
#             pd.concat((oot[("OOT 1", metric_name)], oot[("OOT 1", "% Change")]), axis=1),
#             pd.concat((oot[("OOT 2", metric_name)], oot[("OOT 2", "% Change")]), axis=1),
#         ),
#         axis=1
#     )

#     oot[("Overall", "OOT1 & OOT2 % Change")] = oot[("Overall", "% Change")]
#     del oot[("Overall", "% Change")]

#     style_dict = {
#         ('OOT 1', '% Change'): '{:,.2%}'.format,
#         ('OOT 2', '% Change'): '{:,.2%}'.format,
#         ('Overall', 'OOT1 & OOT2 % Change'): '{:,.2%}'.format,
#     }

#     if metric_format:
#         style_dict[("OOT 1", metric_name)] = f'{metric_format}'.format
#         style_dict[("OOT 2", metric_name)] = f'{metric_format}'.format

#     display(HTML(oot.style.format(style_dict).to_html()))


# def percentage_change_by_metric_by_oot1_and_oot2(df_dict: pd.DataFrame, metric, metric_name: str):
#     d_metric = dict()
#     for k, data_sample in df_dict.items():
#         d_metric[k] = metric(data_sample.TargetOutcome, data_sample.y_pred)

#     df_metric = pd.DataFrame.from_dict(d_metric, orient="index").reset_index().rename(
#         columns={"index": "data_part", 0: metric_name})
#     df_metric[["Model", "DataPart"]] = df_metric.data_part.str.split("|", expand=True)
#     df_metric["Model"] = df_metric["Model"].str.strip()
#     df_metric["DataPart"] = df_metric["DataPart"].str.strip()

#     oot1 = df_metric[df_metric.DataPart == "OOT 1"]
#     oot2 = df_metric[df_metric.DataPart == "OOT 2"]

#     oot1["% Change"] = ((oot1[oot1.Model == "Random Forest New"][metric_name].values[0] - oot1[metric_name]) / oot1[
#         metric_name]).abs().values
#     oot2["% Change"] = ((oot2[oot2.Model == "Random Forest New"][metric_name].values[0] - oot2[metric_name]) / oot2[
#         metric_name]).abs().values

#     return oot1, oot2


# def metric_by_month(unique_months: np.array,
#                     data_prior: pd.DataFrame,
#                     data_new: pd.DataFrame,
#                     metric, metric_name: str,
#                     prior_model_alias: str, new_model_alias: str):
#     l_oot_new = []
#     for date_value in unique_months:
#         temp = data_new[data_new.ModelScoreDate_MonthEnd == date_value]
#         v = metric(temp.TargetOutcome, temp.y_pred)
#         l_oot_new += [v]

#     l_oot_prior = []
#     for date_value in unique_months:
#         temp = data_prior[data_prior.ModelScoreDate_MonthEnd == date_value]
#         v = metric(temp.TargetOutcome, temp.y_pred)
#         l_oot_prior += [v]

#     oot = pd.DataFrame.from_dict(
#         {
#             "ModelScoreDate_MonthEnd": np.concatenate((unique_months, unique_months)),
#             metric_name: l_oot_new + l_oot_prior,
#             "Model": [new_model_alias] * len(unique_months) + [prior_model_alias] * len(unique_months)
#         }
#     )
#     return oot


# def concatenate_oot1_and_oot2(oot1: pd.DataFrame, oot2: pd.DataFrame, metric_name: str):
#     oot = pd.concat((oot1, oot2))
#     oot[metric_name] = oot[metric_name].round(3)
#     oot.ModelScoreDate_MonthEnd = pd.to_datetime(oot.ModelScoreDate_MonthEnd).dt.date
#     oot.sort_values("ModelScoreDate_MonthEnd", ignore_index=True, inplace=True)
#     return oot


# def plot_metric_by_month(oot: pd.DataFrame,
#                          metric_name: str,
#                          palette: dict,
#                          y_annotation: int=1.05,
#                          y_lim: tuple=(0.8, 1.1),
#                          title: str="ROC AUC by months",
#                          y_label: str="ROC AUC"
#                         ):
#     _ = plt.figure(figsize=(10, 5), dpi=150)
#     ax = sns.barplot(
#         x="ModelScoreDate_MonthEnd", y=metric_name, hue="Model", data=oot,
#         palette=palette
#     )
#     i = 0
#     range_n = (0, 3)
#     for model in oot.Model.unique():
#         if i > 1:
#             range_n = (3, 6)
#         ax_ = sns.lineplot(
#             x=np.arange(*range_n), y=metric_name, data=oot[oot.Model == model].reset_index(drop=True),
#             color=palette[model],
#             label=""
#         )
#         i += 1

#     _ = plt.vlines(2.4,  0, y_annotation, colors="grey", linestyles="--")
#     _ = plt.annotate("OOT 1", (0.7, y_annotation), fontsize=15)
#     _ = plt.annotate("OOT 2", (3.85, y_annotation), fontsize=15)

#     for container in ax.containers:
#         ax.bar_label(container)

#     _ = plt.title(title, fontsize=20)
#     _ = plt.ylim(*(y_lim))
#     _ = plt.xlabel("")
#     _ = plt.ylabel(y_label)
#     h,l = ax.get_legend_handles_labels()
#     _ = plt.legend(
#         h[:2], l[:2],
#         loc='upper center', bbox_to_anchor=(0.5, -0.1),
#         fancybox=False, shadow=False, ncol=4
#     )


# def compute_percentage_of_change(oot: pd.DataFrame, metric_name: str):
#     pcnt_change = oot.groupby("ModelScoreDate_MonthEnd").apply(lambda x: (x[x.Model.str.contains("New")][metric_name].values[0] - x[x.Model.str.contains("Prior")][metric_name].values[0]) * 100 / x[x.Model.str.contains("Prior")][metric_name].abs().values[0]).reset_index(name="% change")
#     pcnt_change.ModelScoreDate_MonthEnd = pd.to_datetime(pcnt_change.ModelScoreDate_MonthEnd).dt.date
#     pcnt_change.sort_values("ModelScoreDate_MonthEnd", ignore_index=True, inplace=True)
#     pcnt_change["% change"] = pcnt_change["% change"].round(2)

#     pcnt_change["sign"] = pcnt_change["% change"] < 0
#     return pcnt_change


# def plot_percentage_change(pcnt_change: pd.DataFrame, y_annotation: float=8, y_lim: tuple=(-5, 10), title: str="% change of ROC AUC by months"):
#     _ = plt.figure(figsize=(10, 5), dpi=150)
#     ax = sns.barplot(
#         x="ModelScoreDate_MonthEnd", y="% change", data=pcnt_change, hue="sign", dodge=False,
#         palette={False: "#a0cdfa", True: "red"}
#     )

#     _ = plt.vlines(2.55,  0, y_annotation, colors="grey", linestyles="--")
#     _ = plt.annotate("OOT 1", (0.7, y_annotation), fontsize=15)
#     _ = plt.annotate("OOT 2", (3.85, y_annotation), fontsize=15)

#     for container in ax.containers:
#         ax.bar_label(container)

#     _ = plt.title(title, fontsize=20)
#     _ = plt.legend("")
#     _ = plt.ylim(*y_lim)
#     _ = plt.xlabel("")
#     _ = plt.ylabel("%")


# def precision_recall_auc(y_true: pd.Series, y_pred: pd.Series):
#     pr, r, _  = precision_recall_curve(y_true, y_pred)
#     return auc(r, pr)


# def plot_roc_auc_cv(X: np.array, y: np.array, n_splits: int, random_state: int, n_sigma: int, classifier):
#     cv = StratifiedKFold(n_splits=n_splits)
    
#     tprs = []
#     aucs = []
#     ginis = []
#     mean_fpr = np.linspace(0, 1, 100)

#     for fold, (train, test) in enumerate(cv.split(X, y)):
#         classifier.fit(X[train], y[train])
#         fpr, tpr = roc_curve(y[test], classifier.predict_proba(X[test])[:, 1])[:2]
#         roc_auc_value = auc(fpr, tpr)
#         gini_value = 2 * auc(fpr, tpr) - 1
        
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(roc_auc_value)
#         ginis.append(gini_value)
        
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
    
#     mean_auc = auc(mean_fpr, mean_tpr)
#     mean_gini = 2*auc(mean_fpr, mean_tpr)-1
    
#     std_auc = np.std(aucs)
#     std_gini = np.std(ginis)
    
#     f, ax = plt.subplots(figsize=(20, 5))
    
#     ax.plot(
#         mean_fpr,
#         mean_tpr,
#         color="black",
#         label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, n_sigma*std_auc),
#         lw=2,
#         alpha=0.8,
#     )

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + n_sigma * std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - n_sigma * std_tpr, 0)
#     ax.fill_between(
#         mean_fpr,
#         tprs_lower,
#         tprs_upper,
#         color="grey",
#         alpha=0.2,
#         label=r"$\pm$ {n_sigma} std. dev.".format(n_sigma=n_sigma),
#     )

#     ax.set(
#         xlabel="False Positive Rate",
#         ylabel="True Positive Rate",
#         title=f"Mean ROC curve with variability",
#     )

#     ax.legend(loc="lower right")
#     plt.show()
#     return f, ax, (mean_gini, std_gini)

# def compare_roc_auc_gini(data, label_values, ax, label_col_name, target_name, color_dict):
#     for label_vlaue in label_values:
#         fpr, tpr, thresholds = roc_curve(
#             data[data[label_col_name] == label_vlaue][target_name].reset_index(drop=True), 
#             data[data[label_col_name] == label_vlaue].y_pred.reset_index(drop=True)
#         )
#         _ = ax.plot(fpr, tpr, label=label_vlaue, color=color_dict[label_vlaue])
#     ax.legend(loc="lower right")

# def plot_pr_auc_cv(X: np.array, y: np.array, n_splits: int, random_state: int, n_sigma: int, classifier):
#     cv = StratifiedKFold(n_splits=n_splits)
    
#     precision_valss_vals = []
#     aucs = []
#     mean_recall_vals = np.linspace(0, 1, 100)

#     for fold, (train, test) in enumerate(cv.split(X, y)):
#         classifier.fit(X[train], y[train])
#         precision_vals, recall_vals = precision_recall_curve(y[test], classifier.predict_proba(X[test])[:, 1])[:2]
#         ind = np.argsort(recall_vals)
#         roc_auc_value = auc(recall_vals[ind], precision_vals[ind])
        
#         interp_precision_vals = np.interp(mean_recall_vals, recall_vals[ind], precision_vals[ind])
#         interp_precision_vals[0] = 1
#         precision_valss_vals.append(interp_precision_vals)
#         aucs.append(roc_auc_value)

#     mean_precision_vals = np.mean(precision_valss_vals, axis=0)
#     mean_precision_vals[-1] = 0
#     mean_auc = auc(mean_recall_vals, mean_precision_vals)
    
#     std_auc = np.std(aucs)
    
#     f, ax = plt.subplots(figsize=(20, 5))
    
#     ax.plot(
#         mean_recall_vals,
#         mean_precision_vals,
#         color="black",
#         label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, n_sigma*std_auc),
#         lw=2,
#         alpha=0.8,
#     )

#     std_precision_vals = np.std(precision_valss_vals, axis=0)
#     precision_valss_vals_upper = np.minimum(mean_precision_vals + n_sigma * std_precision_vals, 1)
#     precision_valss_vals_lower = np.maximum(mean_precision_vals - n_sigma * std_precision_vals, 0)
#     ax.fill_between(
#         mean_recall_vals,
#         precision_valss_vals_lower,
#         precision_valss_vals_upper,
#         color="grey",
#         alpha=0.2,
#         label=r"$\pm$ {n_sigma} std. dev.".format(n_sigma=n_sigma),
#     )

#     ax.set(
#         xlabel="Recall",
#         ylabel="Precision",
#         title=f"Mean PR curve with variability",
#     )
#     ax.legend(loc="upper right")
#     return f, ax

def pr_auc(y_true: pd.Series, y_pred: pd.Series) -> float:
    precision_vals, recall_vals = precision_recall_curve(y_true, y_pred)[:2]
    pr_auc_value = auc(recall_vals, precision_vals)
    return pr_auc_value

def compare_pr_auc(data, label_col_name, taget_col_name, ax, color_dict):    
    for value in data[label_col_name].unique():
        precision, recall = precision_recall_curve(
            data[data[label_col_name] == value][taget_col_name].reset_index(drop=True), 
            data[data[label_col_name] == value].y_pred.reset_index(drop=True)
        )[:2]
        sns.lineplot(
            y=precision,
            x=recall,
            color=color_dict[value],
            label=value,
            lw=2,
            alpha=0.8,
            ax=ax
        )
    ax.legend(loc="upper right")