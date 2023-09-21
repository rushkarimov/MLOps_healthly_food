"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def choose_fillna(data: pd.DataFrame) -> pd.DataFrame:
    """
    Заполнение пропусков в DataFrame, в зависимости от типа данных:
    'object' -> 'None',
    'int64' -> 999999,
    'float64' -> 999999,
    'bool' -> False,
    :param data: DataFrame;
    :return DataFrame.
    """
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna('None', inplace=True)
        elif data[column].dtype == 'int64':
            data[column].fillna(999999, inplace=True)
        elif data[column].dtype == 'float64':
            data[column].fillna(999999, inplace=True)
        elif data[column].dtype == 'bool':
            data[column].fillna(False, inplace=True)
    return data


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных:
    :param data: DataFrame;
    :param change_type_columns: словарь с признаками и типами данных;
    :return: DataFrame.
    """
    return data.astype(change_type_columns, errors='raise')


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train:
    :param data: test DataFrame;
    :param unique_values_path: путь до списка с признаками train, для сравнения;
    :return: test DataFrame.
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), 'Разные признаки'
    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями:
    :param drop_columns: список с признаками для удаления;
    :param data: DataFrame;
    :param target_column: целевая переменная;
    :param unique_values_path: путь до файла со словарем;
    :return: None.
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors='ignore'
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, 'w') as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных:
    :param data: DataFrame;
    :param flg_evaluate: флаг для evaluate;
    :return: DataFrame.
    """
    # удаление неиспользуемых колонок
    data = data.drop(kwargs['drop_columns'],axis=1, errors='ignore')
    # проверка DataFrame на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs['unique_values_path']
        )
    else:
        save_unique_train_data(
            data=data,
            drop_columns=kwargs['drop_columns'],
            target_column=kwargs['target_column'],
            unique_values_path=kwargs['unique_values_path'],
        )

    # заполнение пропусков
    data = choose_fillna(data)

    # замена типов данных
    data = transform_types(data=data, change_type_columns=kwargs['change_type_columns_test'])
    return data