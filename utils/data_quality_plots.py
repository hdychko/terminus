import pandas as pd

from typing import List
from typing import Tuple
from typing import Union


import seaborn as sns
import matplotlib.pyplot as plt


def plot_bars_with_trend(data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         label_col: str,
                         title: str,
                         y_lim: tuple,
                         y_label: str,
                         x_label: Union[str, None] = None,
                         palette_col: Union[str, None] = None,
                         caption_font_size: str = 10,
                         trend_color: str = "grey",
                         additional_trend: Union[str, None] = None,
                         additional_trend_label: Union[str, None] = None,
                         additional_trend_captures: bool = False,
                         additional_trend_color: str = "red",
                         annotations_ratio: float = 0.1,
                         figsize: Tuple[float] = (20, 5)
                         ) -> None:
    """
    Plot barplot with its trend using specified columns 
    """
    _ = plt.figure(figsize=figsize, dpi=100)
    palette = data[palette_col].values if palette_col else None
    g = sns.barplot(x=x_col, y=y_col, data=data, palette=palette)
    _ = sns.lineplot(
        x=list(range(len(data))), y=y_col, data=data, color=trend_color
    )

    if additional_trend:
        _ = sns.lineplot(
            x=list(range(len(data))), y=additional_trend, data=data,
            color=additional_trend_color, label=additional_trend_label
        )

    _ = plt.title(title, fontdict=dict(size=20))
    _ = plt.xticks(
        rotation=45, labels=data[label_col].values,
        ticks=list(range(len(data))), fontsize=13
    )
    _ = plt.ylim(*y_lim)
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)

    for index, row in data.iterrows():
        g.text(
            index, row[y_col] + annotations_ratio*row[y_col], f"{row[y_col]}",
            color='black', ha="center", fontdict=dict(size=caption_font_size)
        )

    if (additional_trend is not None) & additional_trend_captures:
        for index, row in data.iterrows():
            g.text(
                index, row[additional_trend] + 0.1 * row[additional_trend],
                f"{row[additional_trend]}",
                color=additional_trend_color, ha="center",
                fontdict=dict(size=caption_font_size)
            )

    _ = plt.vlines(
        2.5, 0, 0.9 * y_lim[1],
        colors="black", linestyles="dashed"
    )
    _ = plt.text(
        0.5, 0.85 * y_lim[1], "Train", fontdict=dict(size=15)
    )
    _ = plt.text(
        3.5, 0.85 * y_lim[1], "Test 1", fontdict=dict(size=15)
    )

    # _ = plt.vlines(
    #     14.5, y_lim[0], 0.9 * y_lim[1],
    #     colors="black", linestyles="dashed"
    # )
    # _ = plt.text(
    #     3.5, 0.9 * y_lim[1], "Test 2", fontdict=dict(size=15)
    # )
    plt.show()


def plot_stacked_barplot_with_its_trend(data: pd.DataFrame,
                                        cols_names: list,
                                        trend_col: str,
                                        x_label: str, y_label: str,
                                        title: str,
                                        y_lim: tuple,
                                        colors: List[str] = ["#17264D", "#a0cdfa", "#2EE86A"],
                                        trend_color: str = "#17264D",
                                        figsize: Tuple[float]=(20, 5)):
    """
    Plot stacked barplot with its trend
    """
    data[cols_names].plot(
        kind='bar', stacked=True, color=colors, figsize=figsize,
        colormap="Set2"
    )
    if trend_col:
        _ = sns.lineplot(
            x=list(range(len(data))), y=trend_col, data=data, 
            color=trend_color
        )

    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.title(title, fontdict={"size": 20})
    _ = plt.xticks(fontsize=12, rotation=45)
    _ = plt.vlines(
        2.5, 0, 1.1 * y_lim[1],
        colors="black", linestyles="dashed"
    )
    _ = plt.text(
        0.5, 0.9 * y_lim[1], "Train", fontdict=dict(size=15)
    )
    _ = plt.text(
        3.5, 0.9 * y_lim[1], "Test 1", fontdict=dict(size=15)
    )

    # _ = plt.vlines(
    #     11.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1], 
    #     colors="black", linestyles="dashed"
    # )
    # _ = plt.text(
    #     4.5, y_lim[1] - 0.15 * y_lim[1], "Train", fontdict=dict(size=15)
    # )
    # _ = plt.text(
    #     13, y_lim[1] - 0.15 * y_lim[1], "Test", fontdict=dict(size=15)
    # )

    # _ = plt.vlines(
    #     14.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1],
    #     colors="black", linestyles="dashed"
    # )
    # _ = plt.text(
    #     16, y_lim[1] - 0.15 * y_lim[1], "Test 2", fontdict=dict(size=15)
    # )

    leg = plt.legend(loc=(1.01, 0.8))
    plt.show()


def plot_bar_per_group(data: pd.DataFrame,
                       x_col: str, y_col: str, hue_col: str,
                       color_dict: dict,
                       y_lim: tuple,
                       x_label: str, y_label: str, title: str,
                       legend_loc: Tuple[float] = (1.01, 0.65),
                       figsize: Tuple[float] = (20, 5)):
    """
    Plot bars per group with their trends
    """
    _ = plt.figure(figsize=figsize, dpi=100)

    _ = sns.barplot(
        x=x_col, y=y_col, data=data, hue=hue_col, palette=color_dict
        )
    _ = sns.lineplot(
        x=list(range(data[x_col].nunique())) * 3,
        y=y_col,
        data=data,
        hue=hue_col, palette=color_dict
    )

    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.title(title, fontdict={"size": 20})
    _ = plt.xticks(fontsize=12, rotation=45)

    _ = plt.vlines(
        2.5, 0, 1.1 * y_lim[1],
        colors="black", linestyles="dashed"
    )
    _ = plt.text(
        0.5, 0.9 * y_lim[1], "Train", fontdict=dict(size=15)
    )
    _ = plt.text(
        3.5, 0.9 * y_lim[1], "Test 1", fontdict=dict(size=15)
    )

    # _ = plt.vlines(11.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1], colors="black", linestyles="dashed")
    # _ = plt.text(4.5,  y_lim[1] - 0.15 * y_lim[1], "Train", fontdict=dict(size=15))
    # _ = plt.text(13, y_lim[1] - 0.15 * y_lim[1], "Test", fontdict=dict(size=15))
    
    # _ = plt.vlines(14.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1], colors="black", linestyles="dashed")
    # _ = plt.text(16, y_lim[1] - 0.15 * y_lim[1], "Test 2", fontdict=dict(size=15))

    _ = plt.legend(loc=legend_loc)
    plt.show()


def plot_lines(data: pd.DataFrame,
               x_col: str, y_cols: list,
               dict_colors: dict,
               x_label: str, y_label: str, title: str,
               y_lim: tuple,
               figsize: Tuple[float] = (20, 5),
               legend_loc: Tuple[float] = (1.01, 0.5)):
    """
    Plot line per group
    """
    x = range(data.shape[0])
    f = plt.figure(figsize=figsize, dpi=100)
    for col in y_cols: 
        _ = sns.lineplot(
            x=x, y=col, data=data.sort_values(x_col).reset_index(drop=True),
            color=dict_colors[col], label=col
        )

    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.title(title, fontdict={"size": 20})
    _ = plt.xticks(labels=data[x_col], ticks=x, fontsize=12, rotation=45)
    _ = plt.vlines(
        2.5, 0, 1.1 * y_lim[1],
        colors="black", linestyles="dashed"
    )
    _ = plt.text(
        0.5, 0.9 * y_lim[1], "Train", fontdict=dict(size=15)
    )
    _ = plt.text(
        3.5, 0.9 * y_lim[1], "Test 1", fontdict=dict(size=15)
    )
    _ = plt.legend(loc=legend_loc)
    # _ = plt.vlines(11.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1], colors="black", linestyles="dashed")
    # _ = plt.text(4.5,  y_lim[1] - 0.15 * y_lim[1], "Train", fontdict=dict(size=15))
    # _ = plt.text(13, y_lim[1] - 0.15 * y_lim[1], "Test", fontdict=dict(size=15))
    
    # _ = plt.vlines(14.5, y_lim[0], y_lim[1] - 0.10 * y_lim[1], colors="black", linestyles="dashed")
    # _ = plt.text(16, y_lim[1] - 0.15 * y_lim[1], "Test 2", fontdict=dict(size=15))
    plt.show()


def print_stats(data: pd.DataFrame, target_col: str) -> None:
    n_records: int = data.shape[0]
    n_pos_records: int = data[target_col].sum()
    n_neg_records: int = (data[target_col] == 0).sum()
    print("#total records: ", n_records)
    print(
        "#positive class: ",
        n_pos_records, 
        "({}%)".format(n_pos_records * 100 / n_records)
    )
    print(
        "#negative class: ",
        n_neg_records, 
        "({}%)".format(n_neg_records * 100 / n_records)
    )
    print("\n")

    n_users = data.ACCNO.nunique()
    n_positive_users = data.groupby("ACCNO")[target_col].apply(
        lambda x: any(x == 1)
        ).sum()
    n_negative_users = data.groupby("ACCNO")[target_col].apply(
        lambda x: all(x == 0)
        ).sum()
    print("#Total users: ", n_users)
    print(
        "#positive class: ", n_positive_users, 
        "({}%)".format(n_positive_users * 100 / n_users)
    )
    print(
        "#negative class: ", n_negative_users, 
        "({}%)".format(n_negative_users * 100 / n_users)
    )
