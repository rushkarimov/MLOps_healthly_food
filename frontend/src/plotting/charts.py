"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def classic_barplot(data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    title: str,
                    palette='husl',
                    xticks_rotation=0,
                    xticks_fontsize=12,
                    yticks_fontsize=12) -> None:
    """"
    Создание индивидуального графика barplot:
    :param data: DataFrame;
    :param x_col: значения по оси x;
    :param y_col: значения по оси y;
    :param title: наименование графика;
    :param palette: цвет графика;
    :param xticks_rotation: угол поворота обозначений по оси x;
    :param xticks_fontsize: размер обозначений по оси x;
    :param yticks_fontsize: размер обозначений по оси y;
    :return None.
    """
    fig = plt.figure(figsize=(15, 7))
    sns.barplot(x=x_col, y=y_col, data=data, palette=palette)

    plt.title(title, fontsize=18)
    plt.xlabel(x_col, fontsize=16)
    plt.ylabel(y_col, fontsize=16)

    plt.xticks(rotation=xticks_rotation)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.show()
    return fig


def classic_boxplot(data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    title: str,
                    palette='husl',
                    xticks_rotation=0,
                    xticks_fontsize=12,
                    yticks_fontsize=12) -> None:
    """"
    Создание индивидуального графика boxplot:
    :param data: DataFrame;
    :param x_col: значения по оси x;
    :param y_col: значения по оси y;
    :param title: наименование графика;
    :param palette: цвет графика;
    :param xticks_rotation: угол поворота обозначений по оси x;
    :param xticks_fontsize: размер обозначений по оси x;
    :param yticks_fontsize: размер обозначений по оси y;
    :return None.
    """
    fig = plt.figure(figsize=(15, 7))
    sns.boxplot(x=x_col, y=y_col, data=data, palette=palette)

    plt.title(title, fontsize=18)
    plt.xlabel(x_col, fontsize=16)
    plt.ylabel(y_col, fontsize=16)

    plt.xticks(rotation=xticks_rotation)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.show()
    return fig


def classic_violinplot(data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str,
                       palette='husl',
                       xticks_rotation=0,
                       xticks_fontsize=12,
                       yticks_fontsize=12) -> None:
    """"
    Создание индивидуального графика violinplot:
    :param data: DataFrame;
    :param x_col: значения по оси x;
    :param y_col: значения по оси y;
    :param title: наименование графика;
    :param palette: цвет графика;
    :param xticks_rotation: угол поворота обозначений по оси x;
    :param xticks_fontsize: размер обозначений по оси x;
    :param yticks_fontsize: размер обозначений по оси y;
    :return None.
    """
    fig = plt.figure(figsize=(15, 7))
    sns.violinplot(x=x_col, y=y_col, data=data, palette=palette)

    plt.title(title, fontsize=18)
    plt.xlabel(x_col, fontsize=16)
    plt.ylabel(y_col, fontsize=16)

    plt.xticks(rotation=xticks_rotation)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.show()
    return fig