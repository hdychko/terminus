from typing import Any
from typing import Tuple
from typing import List
from typing import Union

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



def compute_freqs(data: pd.DataFrame, col_name: str, target_col: str):
    temp = data.groupby(["DataPart", target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp = pd.merge(temp, temp.groupby(["DataPart", target_col]).n.sum().reset_index(name="total"), how="left", on=["DataPart", target_col])
    temp["%"] = temp["n"] * 100 / temp["total"]
    return temp

def compute_freqs_by_month(data: pd.DataFrame, col_name: str, target_col: str, month_col: str, data_part_col: str="DataPart") -> pd.DataFrame:
    temp = data.groupby([month_col, target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp = pd.merge(temp, temp.groupby([month_col, target_col]).n.sum().reset_index(name="total"), how="left", on=[month_col, target_col])
    temp = pd.merge(
        temp, 
        data.groupby([month_col]).apply(lambda x: x[data_part_col].unique()[0]).reset_index(name=data_part_col), 
        how="left", on=[month_col]
    )
    temp["%"] = temp["n"] * 100 / temp["total"]
    return temp


def plot_train_test_fature_distr(data: pd.DataFrame,
                                 col_name: str, 
                                 target_col: str,
                                 color_map=None, 
                                 legend_loc=None, 
                                 figsize: Tuple[float]=(20, 5)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` in train/test datasets', fontsize=20)

    p1 = sns.barplot(
        x=target_col, y="%", hue=col_name, data=data[data.DataPart == "train"], 
        ax=ax[0], palette=color_map
    )
    _ = ax[0].set_title('Train')

    for p in p1.patches:
        if p.get_height() == 0:
            continue
        p1.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)
    _ = ax[0].legend(loc=legend_loc)

    p2 = sns.barplot(
        x=target_col, y="%", hue=col_name, data=data[data.DataPart == "test"], 
        ax=ax[1], palette=color_map
    )
    for p in p2.patches:
        if p.get_height() == 0:
            continue
        p2.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)


    _ = ax[1].set_title('Test')
    _ = ax[1].legend(loc=legend_loc)
    plt.show()
    
def plot_fature_distr_by_month(data: pd.DataFrame,
                              col_name: str, 
                              target_col: str,
                              month_col: str,
                              target_val: int=1,
                              col_vals: Union[List, None]=None,
                              color_map=None, 
                              legend_loc=None, 
                              figsize: Tuple[float]=(20, 5), 
                              data_parts: Union[List, None]=None):
    data_temp = data[data[target_col] == target_val].reset_index(drop=True).copy()
    if col_vals is not None:
        data_temp = data[data[col_name].isin(col_vals)].reset_index(drop=True)
    if data_parts is not None:
        data_temp = data[data.DataPart.isin(data_parts)].reset_index(drop=True)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` by month. {data_parts}', fontsize=20)

    p1 = sns.pointplot(
        x=month_col, y="%", hue=col_name, data=data_temp, 
        ax=ax, palette=color_map
    )
    _ = ax.legend(loc=legend_loc, title=col_name)

    plt.show()


def plot_train_test_continues_fature_distr(data: pd.DataFrame,
                                           col_name: str,
                                           target_col: str,
                                           legend_loc=None,
                                           if_patches: bool=True,
                                           figsize: Tuple[float]=(20, 5)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` in train/test datasets', fontsize=20)

    p1 = sns.boxplot(
        y=target_col, x=col_name, data=data[data.DataPart.map(lambda x : "train" in x)], 
        ax=ax[0], orient="h"
    )
    _ = ax[0].set_title('Train')
    if if_patches:
        for p in p1.patches:
            if p.get_height() == 0:
                continue
            p1.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)
    _ = plt.legend(loc=legend_loc)
    _ = ax[0].legend(loc=legend_loc)

    p2 = sns.boxplot(
        y=target_col, x=col_name, data=data[data.DataPart.map(lambda x : "test" in x)], 
        ax=ax[1], orient="h"
    )
    if if_patches:
        for p in p2.patches:
            if p.get_height() == 0:
                continue
            p2.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)
    _ = ax[1].legend(loc=legend_loc)

    _ = ax[1].set_title('Test')
    plt.show()


def plot_continues_feat_overall(data: pd.DataFrame,
                                col_name,
                                target_col: str,
                                xlim=None,
                                legend_loc=None,
                                if_patches=True,
                                figsize: Tuple[float]=(20, 5)):
    _ = plt.figure(figsize=figsize, dpi=250)
    pp = sns.boxplot(y=target_col, x=col_name, data=data, orient="h")
    if if_patches:
        for p in pp.patches:
            if p.get_height() == 0:
                continue
            pp.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)
    if xlim is not None:
        _ = plt.title(f"Distribution of `{col_name}` by Target. Zoomed", fontdict={"size": 20})
    else:
        _ = plt.title(f"Distribution of `{col_name}` by Target", fontdict={"size": 20})
    _ = plt.xlim(xlim)
    _ = plt.legend(loc=legend_loc)
    plt.show()


def plot_feat_overall(data: pd.DataFrame,
                      col_name: str,
                      target_col: str,
                      color_map=None, 
                      legend_loc=None,
                      figsize: Tuple[float]=(20, 5)):
    _ = plt.figure(figsize=figsize, dpi=250)
    temp = data.groupby([target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp = pd.merge(temp, data.groupby([target_col]).ACCNO.count().reset_index(name="total"), how='left', on=target_col)
    temp["%"] = temp["n"] * 100 / temp["total"]
    temp.reset_index(drop=True, inplace=True)
    pp = sns.barplot(x=target_col, y="%", hue=col_name, data=temp, palette=color_map)

    for p in pp.patches:
        if p.get_height() == 0:
            continue
        pp.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)

    _ = plt.title(f"Distribution of `{col_name}` by Target", fontdict={"size": 20})
    _ = plt.legend(loc=legend_loc)
    plt.show()


def compute_freqs_for_users(data: pd.DataFrame,
                            col_name: str,
                            target_col: str):
    
    temp_test = data[data.DataPart.map(lambda x: "test" in x)].groupby([target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp_test = pd.merge(temp_test, temp_test.groupby(target_col).n.sum().reset_index(name="total"), how="left", on=target_col)
    temp_test["%"] = temp_test.n * 100 / temp_test.total
    temp_test["DataPart"] = "test"
    
    temp_train = data[data.DataPart.map(lambda x: "train" in x)].groupby([target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp_train = pd.merge(temp_train, temp_train.groupby(target_col).n.sum().reset_index(name="total"), how="left", on=target_col)
    temp_train["%"] = temp_train.n * 100 / temp_train.total
    temp_train["DataPart"] = "train"
    
    temp = pd.concat((temp_test, temp_train)).reset_index(drop=True)
    return temp

def compute_freqs_by_bt_for_users(data: pd.DataFrame, 
                                  col_name: str,
                                  target_col: str):
    temp_test = data[data.DataPart.map(lambda x: "test" in x)].groupby(["BusinessType", target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp_test = pd.merge(temp_test, temp_test.groupby(["BusinessType", target_col]).n.sum().reset_index(name="total"), how="left", on=["BusinessType", target_col])
    temp_test["%"] = temp_test.n * 100 / temp_test.total
    temp_test["DataPart"] = "test"
    
    temp_train = data[data.DataPart.map(lambda x: "train" in x)].groupby(["BusinessType", target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp_train = pd.merge(temp_train, temp_train.groupby(["BusinessType", target_col]).n.sum().reset_index(name="total"), how="left", on=["BusinessType", target_col])
    temp_train["%"] = temp_train.n * 100 / temp_train.total
    temp_train["DataPart"] = "train"
    
    temp = pd.concat((temp_test, temp_train)).reset_index(drop=True)

    return temp


def compute_freqs_by_bt(data: pd.DataFrame, 
                        col_name: str, 
                        target_col: str):
    temp = data.groupby(["DataPart", "BusinessType", target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp = pd.merge(temp, temp.groupby(["DataPart", "BusinessType", target_col]).n.sum().reset_index(name="total"), how="left", on=["DataPart", "BusinessType", target_col])
    temp["%"] = temp["n"] * 100 / temp["total"]
    return temp

def compute_freqs_by_bt_by_month(data: pd.DataFrame, 
                                 col_name: str, 
                                 target_col: str, 
                                 month_col: str):
    
    temp = data.groupby(["BusinessType", month_col, target_col, col_name]).ACCNO.count().reset_index(name="n")
    temp = pd.merge(temp, temp.groupby(["BusinessType", month_col, target_col]).n.sum().reset_index(name="total"), how="left", on=[month_col, "BusinessType", target_col])
    temp["%"] = temp["n"] * 100 / temp["total"]   
    return temp

def plot_train_test_continues_fature_distr_by_bt(data: pd.DataFrame, 
                                                 col_name: str, 
                                                 target_col: str,
                                                 legend_loc=None, 
                                                 if_patches=True):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` in train/test datasets', fontsize=20)

    for i, business_type in enumerate(["Consumer", "Sole.Trader", "Limited.Company"]):
        if not any(data.BusinessType==business_type):
            continue
        p1 = sns.boxplot(
            y=target_col, x=col_name, data=data[(data.DataPart.map(lambda x: "train" in x)) & (data.BusinessType==business_type)], 
            ax=ax[i, 0],orient="h"
        )
        _ = ax[i, 0].set_title("Train, " + business_type)
        _ = ax[i, 0].legend(loc=legend_loc)
        if if_patches:
            for p in p1.patches:
                if p.get_height() == 0:
                    continue
                p1.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)

        p2 = sns.boxplot(
            y=target_col, x=col_name, data=data[(data.DataPart.map(lambda x: "test" in x)) & (data.BusinessType==business_type)], 
            ax=ax[i, 1],orient="h"
        )
        if if_patches:
            for p in p2.patches:
                if p.get_height() == 0:
                    continue
                p2.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)


        _ = ax[i, 1].set_title("Test, " + business_type)
        _ = ax[i, 1].legend(loc=legend_loc)


def plot_train_test_fature_distr_by_bt(data: pd.DataFrame, 
                                       col_name: str, 
                                       target_col: str,
                                       color_map=None, 
                                       legend_loc=None):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` in train/test datasets', fontsize=20)

    for i, business_type in enumerate(["Consumer", "Sole.Trader", "Limited.Company"]):
        p1 = sns.barplot(
            x=target_col, y="%", hue=col_name, data=data[(data.DataPart == "train") & (data.BusinessType==business_type)], 
            ax=ax[i, 0], palette=color_map
        )
        _ = ax[i, 0].set_title("Train, " + business_type)
        
        for p in p1.patches:
            if p.get_height() == 0:
                continue
            p1.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)
        _ = ax[i, 0].legend(loc=legend_loc)
        
        p2 = sns.barplot(
            x=target_col, y="%", hue=col_name, data=data[(data.DataPart == "test") & (data.BusinessType==business_type)], 
            ax=ax[i, 1], palette=color_map
        )
        for p in p2.patches:
            if p.get_height() == 0:
                continue
            p2.annotate("%.0f%%" % p.get_height(), ((p.get_x()+p.get_width()/2), p.get_height()+2), ha="center", va="center", fontsize=15)


        _ = ax[i, 1].set_title("Test, " + business_type)
        _ = ax[i, 1].legend(loc=legend_loc)
        

def plot_fature_distr_by_bt_by_month(data: pd.DataFrame, 
                                    col_name: str, 
                                    target_col: str,
                                    month_col: str,
                                    col_values: Union[List, None]=None,
                                    target_val: Union[List, None]=None,
                                    color_map=None, 
                                    legend_loc=None, 
                                    data_parts: Union[List, None]=None) -> None:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 20), sharex=False, sharey=True)
    fig.suptitle(f'Distribution of `{col_name}` by month', fontsize=20)
    
    data_cp = data[data[target_col] == target_val].reset_index(drop=True).copy()
    
    if data_parts is not None:
        data_cp = data_cp[data_cp.DataPart.isin(data_parts)].reset_index(drop=True)
        
    if col_values is not None:
        data_cp = data_cp[data_cp[col_name].isin(col_values)].reset_index(drop=True)

    for i, business_type in enumerate(["Consumer", "Sole.Trader", "Limited.Company"]):
        p1 = sns.pointplot(
            x=month_col, y="%", hue=col_name, data=data_cp[(data_cp.BusinessType==business_type)], 
            ax=ax[i], palette=color_map
        )
        _ = ax[i].set_title(business_type)
        _ = ax[i].legend(loc=legend_loc, title=col_name)
    plt.show()
