#!/usr/bin/env python
# coding: utf-8
from typing import List
from typing import Any
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import auc


def first_payment_date_id_pred_as_fraud(data: pd.DataFrame,
                                        cut_off: float, 
                                        y_pred_col_name: str) -> pd.DataFrame:
    df_pred = data[data[y_pred_col_name] > cut_off]\
        .groupby('CustomerNumber')\
        .PaymentDateTime.first()\
        .reset_index(name='PredDateTime')
    
    df_pred = pd.merge(
        df_pred, 
        data[data[y_pred_col_name] > cut_off]\
            .groupby('CustomerNumber')\
            .PaymentId.first()\
            .reset_index(name='PredPaymentId'),
        how='left', on='CustomerNumber'
    )
    return df_pred


def first_payment_date_id_act_as_fraud(data: pd.DataFrame,
                                       y_true_col_name: str,
                                       y_true_values: List[Any]) -> pd.DataFrame:
    df_pred = data[data[y_true_col_name].isin(y_true_values)]\
        .groupby('CustomerNumber')\
        .PaymentDateTime.first()\
        .reset_index(name='ActDateTime')

    df_pred = pd.merge(
        df_pred, 
        data[data[y_true_col_name].isin(y_true_values)]\
            .groupby('CustomerNumber')\
            .PaymentId.first()\
            .reset_index(name='ActPaymentId'),
        how='left', on='CustomerNumber'
    )
    df_pred = pd.merge(
        df_pred, 
        data[data[y_true_col_name].isin(y_true_values)]\
            .groupby('CustomerNumber')[y_true_col_name].first()\
            .reset_index(name='First' + y_true_col_name),
        how='left', on='CustomerNumber'
    )
    return df_pred


def first_pred_act_target(data: pd.DataFrame, 
                          cut_off: float,
                          y_pred_col_name: str,
                          y_true_col_name: str,
                          y_true_values: List[Any]
                         ) -> pd.DataFrame:
    df_pred = first_payment_date_id_pred_as_fraud(
        data=data, cut_off=cut_off, y_pred_col_name=y_pred_col_name
    )
    
    df_act = first_payment_date_id_act_as_fraud(
        data=data, y_true_col_name=y_true_col_name, y_true_values=y_true_values
    )
    df_full = pd.merge(df_pred, df_act, on='CustomerNumber', how='outer')
    
    for col in [
        'ActDateTime', 'ActPaymentId', 
        'PredDateTime', 'PredPaymentId', 
        'First' + y_true_col_name
    ]:
        if col in data.columns:
            del data[col]
    
    data = pd.merge(data, df_full, on='CustomerNumber', how='left')
    return data


def fp(data: pd.DataFrame, 
       cut_off: float,
       y_pred_col_name: str) -> pd.DataFrame:
    
    mask = ~data.PredDateTime.isnull() & data.ActDateTime.isnull()
    data_reached_cutoff = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId', y_pred_col_name]
    ].copy()
    
    ids = data_reached_cutoff[
        data_reached_cutoff[y_pred_col_name] > cut_off
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'FP'
    return data


def tp(data: pd.DataFrame) -> pd.DataFrame:
    # TP
    mask = ~data.PredDateTime.isnull() & ~data.ActDateTime.isnull()
    data_act_ones = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId']
    ].copy()
    
    # orrectly classified the first marked payment
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime == data_act_ones.PredDateTime) & 
        (data_act_ones.PaymentDateTime == data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TP'
    del ids
    
    # if predicted payment as a fraud is before act - everything after predicted is TP
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime >= data_act_ones.PredDateTime) & 
        (data_act_ones.PredDateTime < data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TP'
    del ids
    
    # if predicted payment as a fraud is after act - everything after act is TP    
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime >= data_act_ones.PredDateTime) & 
        (data_act_ones.PredDateTime >= data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TP'
    del ids
    return data


def tn(data: pd.DataFrame,
       cut_off: float,
       y_pred_col_name: str) -> pd.DataFrame:
    # TN
    mask = data.PredDateTime.isnull() & data.ActDateTime.isnull()
    data.loc[mask, 'Error'] = 'TN'
    
    mask = ~data.PredDateTime.isnull() & data.ActDateTime.isnull()
    data_reached_cutoff = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId', y_pred_col_name]
    ].copy()
    ids = data_reached_cutoff[
        data_reached_cutoff[y_pred_col_name] <= cut_off
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TN'
    del ids
    
#     ids = data_reached_cutoff[
#         (data_reached_cutoff.PaymentDateTime - data_reached_cutoff.PredDateTime) / np.timedelta64(1, 'h') > 0 &
#         (data_reached_cutoff.PaymentDateTime - data_reached_cutoff.PredDateTime) / np.timedelta64(1, 'h') < 720
#     ].PaymentId.values
#     data.loc[data.PaymentId.isin(ids), 'Error'] = 'TN'
#     del ids
    
    mask = ~data.PredDateTime.isnull() & ~data.ActDateTime.isnull()
    data_act_ones = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId']
    ].copy()
    
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime < data_act_ones.PredDateTime) & 
        (data_act_ones.PredDateTime <= data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TN'
    del ids
    
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime < data_act_ones.ActDateTime) & 
        (data_act_ones.ActDateTime <= data_act_ones.PredDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TN'
    
    
    mask = data.PredDateTime.isnull() & ~data.ActDateTime.isnull()
    data_act_ones = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId']
    ].copy()
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime < data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'TN'
    return data


def fn(data: pd.DataFrame) -> pd.DataFrame:
    mask = ~data.PredDateTime.isnull() & ~data.ActDateTime.isnull()
    data_act_ones = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId']
    ].copy()
    
    # FN
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime > data_act_ones.ActDateTime) & 
        (data_act_ones.ActDateTime < data_act_ones.PredDateTime) & 
        (data_act_ones.PaymentDateTime < data_act_ones.PredDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'FN'
    del ids
    
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime == data_act_ones.ActDateTime) & 
        (data_act_ones.ActDateTime < data_act_ones.PredDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'FN'
    del ids

    mask = data.PredDateTime.isnull() & ~data.ActDateTime.isnull()
    data_act_ones = data.loc[
        mask, 
        ['PaymentDateTime', 'PredDateTime', 'ActDateTime', 'PaymentId']
    ].copy()
    ids = data_act_ones[
        (data_act_ones.PaymentDateTime >= data_act_ones.ActDateTime)
    ].PaymentId.values
    data.loc[data.PaymentId.isin(ids), 'Error'] = 'FN'
    return data


def error_type(data: pd.DataFrame, 
               cut_off: float,
               y_pred_col_name: str) -> pd.DataFrame:
    dict_to_check = dict()
    data['Error'] = None
    
    # FP
    data = fp(data, cut_off, y_pred_col_name)
    dict_to_check['FP'] = set(data[data.Error.isin(['FP'])].PaymentId.values)
  
    # TP
    data = tp(data)    
    dict_to_check['TP'] = set(data[data.Error.isin(['TP'])].PaymentId.values)
      
    # TN
    data = tn(data, cut_off, y_pred_col_name)
    dict_to_check['TN'] = set(data[data.Error.isin(['TN'])].PaymentId.values)
    
    # FN
    data = fn(data)
    dict_to_check['FN'] = set(data[data.Error.isin(['FN'])].PaymentId.values)
    
    for k, v in dict_to_check.items():
        sub_dict = {kk: vv for kk, vv in dict_to_check.items() if kk != k}
        for kk, vv in sub_dict.items():
            inter = vv.intersection(v)
            if inter != set():
                print(f'{k}--{kk}: ', vv.intersection(v))
                raise ValueError('Types of errors can\'t intersect')
    
    assert not data.Error.isnull().any(), f'Nulls in errors: e.g. {data[data.Error.isnull()].PaymentId.values[:1]}'
    return data


def precision_func(data: pd.DataFrame) -> float:
    stats = data.to_dict()
    return stats.get('TP', 0) / (stats.get('TP', 0) +  stats.get('FP', 0))

def recall_func(data: pd.DataFrame) -> float:
    stats = data.to_dict()
    return stats.get('TP', 0) / (stats.get('TP', 0) + stats.get('FN', 0))


def cumulative_analytics(data: pd.DataFrame,
                         n_quantiles: int,
                         y_pred_col_name: str, 
                         y_true_col_name: str,
                         y_true_values: List[Any],
                         decimals: int = 3
                         ) -> Tuple[pd.DataFrame]:

    quantiles = pd.qcut(data[y_pred_col_name].round(decimals), n_quantiles, precision=decimals)
    data[f'Q{n_quantiles}_{y_pred_col_name}'] = quantiles
    
    quantiles_pos = data.groupby(
        quantiles
    )[y_true_col_name].apply(lambda x: sum(x.isin(y_true_values))).reset_index(name='n_pos')

    df_stats = pd.DataFrame(quantiles.cat.categories, columns=['Bins'])
    df_stats = pd.merge(
        df_stats, pd.DataFrame(quantiles.value_counts()).rename(columns={y_pred_col_name: '# payments', 'count': '# payments'}), 
        how='outer', left_on=['Bins'], right_index=True
    )
    df_stats = pd.merge(df_stats, quantiles_pos, left_on=['Bins'], right_on=[y_pred_col_name], how='outer')
    
    df_stats['Bins_l'] = df_stats['Bins'].apply(lambda x: float(x.left))
    df_stats['Bins_r'] = df_stats['Bins'].apply(lambda x: float(x.right))
    unique_bins = tuple(zip(df_stats['Bins_l'].values, df_stats['Bins_r'].values))

    df_stats.set_index(['Bins_l', 'Bins_r'], inplace=True)

    df_stats['Cumulative Precision'] = None
    df_stats['Cumulative Recall'] = None

    df_stats['FP'] = None
    df_stats['TP'] = None

    df_stats['FN'] = None
    df_stats['TN'] = None


    df_stats['FP_prior'] = None
    df_stats['TP_prior'] = None

    df_stats['FN_prior'] = None
    df_stats['TN_prior'] = None

    for cut_off_l, cut_off_r in unique_bins:
        data = first_pred_act_target(
            data=data, 
            cut_off=cut_off_l,
            y_pred_col_name=y_pred_col_name,
            y_true_col_name=y_true_col_name,
            y_true_values=y_true_values
        )
        data = error_type(data=data, cut_off=cut_off_l, y_pred_col_name=y_pred_col_name)
        stats = data.Error.value_counts()

        cum_precision = precision_func(stats) * 100
        cum_recall = recall_func(stats) * 100

        df_stats.loc[(cut_off_l, cut_off_r), 'Cumulative Precision'] = round(cum_precision, 2)
        df_stats.loc[(cut_off_l, cut_off_r), 'Cumulative Recall'] = round(cum_recall, 2)

        for error in ['TP', 'FP', 'TN', 'FN']:
            df_stats.loc[(cut_off_l, cut_off_r), error] = int(stats.get(error, 0))

        df_stats.loc[(cut_off_l, cut_off_r), 'FP_prior'] = (
            (data[y_pred_col_name].round(decimals) > cut_off_l) & 
            (~data[y_true_col_name].isin(y_true_values))
        ).sum()
        df_stats.loc[(cut_off_l, cut_off_r), 'TP_prior'] = (
            (data[y_pred_col_name].round(decimals) > cut_off_l) & 
            data[y_true_col_name].isin(y_true_values)
        ).sum()
        df_stats.loc[(cut_off_l, cut_off_r), 'TN_prior'] = (
            (data[y_pred_col_name].round(decimals) <= cut_off_l) & 
            (~data[y_true_col_name].isin(y_true_values))
        ).sum()
        df_stats.loc[(cut_off_l, cut_off_r), 'FN_prior'] = (
            (data[y_pred_col_name].round(decimals) <= cut_off_l) & 
            data[y_true_col_name].isin(y_true_values)
        ).sum()

        df_stats.loc[(cut_off_l, cut_off_r), 'Total Positive'] = df_stats.loc[(cut_off_l, cut_off_r), 'TP'] + \
            df_stats.loc[(cut_off_l, cut_off_r), 'FN']

    df_stats = df_stats.sort_index(level=1, ascending=False)

    df_stats.reset_index(drop=True, inplace=True)
    
    df_stats['% payments'] = round(df_stats['# payments'] * 100 / df_stats['# payments'].sum(), 2)
    df_stats['% frauds'] = round(df_stats['n_pos'] * 100 / (df_stats['n_pos'].sum()), 2)
    
    df_stats['Cum # payments'] = df_stats['# payments'].cumsum().astype(int)
    df_stats['CumP_prior'] = df_stats['n_pos'].cumsum().astype(int)

    df_stats['Cum Precision_prior'] = round(df_stats.CumP_prior * 100 / (df_stats['Cum # payments']), 2)
    df_stats['Cum Recall_prior'] = round(df_stats.CumP_prior * 100 / (df_stats['n_pos'].sum()), 2)
    return df_stats, data 

def modified_pr_rec_auc(data: pd.DataFrame,
                         y_pred_col_name: str, 
                         y_true_col_name: str,
                         y_true_values: List[Any]
                         ) -> Tuple[pd.DataFrame]:

    thresholds = np.sort(np.unique(data[y_pred_col_name]))
    precision_list = [
        data[y_true_col_name].isin(y_true_values).sum() / data.shape[0]
    ]
    recall_list = []
    for cut_off in thresholds:
        data = first_pred_act_target(
            data=data, 
            cut_off=cut_off,
            y_pred_col_name=y_pred_col_name,
            y_true_col_name=y_true_col_name,
            y_true_values=y_true_values
        )
        data = error_type(data=data, cut_off=cut_off, y_pred_col_name=y_pred_col_name)
        stats = data.Error.value_counts()

        cum_precision = precision_func(stats)
        cum_recall = recall_func(stats)
        
        precision_list += [cum_precision]
        recall_list += [cum_recall]
    
    recall_list += [1]
    pr_recall = auc(x=recall_list, y=precision_list)
    return pr_recall
