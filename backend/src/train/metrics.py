"""
Программа: Получение метрик
Версия: 1.0
"""
import json

import yaml
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import pandas as pd
import numpy as np


def create_dict_metrics(y_test: np.ndarray, y_pred: np.array,
                           X_test: np.ndarray) -> dict:
    """
    Получение словаря с метриками для задачи регрессии и запись в словарь:
    :param y_test: реальные значения;
    :param y_pred: предсказанные значения;
    :param X_test: признаки для предсказания;
    :return: dict.
    """
    dict_metrics = {
        'MAE':
        round(mean_absolute_error(y_test, y_pred), 3),
        'MSE':
        round(mean_squared_error(y_test, y_pred), 3),
        'RMSE':
        round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        'R2_adjusted':
        round(
            1 - (1 - r2_score(y_test, y_pred)) *
            ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)), 3),
        'WAPE':
        round(np.sum(np.abs(y_pred - y_test)) / np.sum(y_test) * 100, 3)
    }

    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик:
    :param data_x: объект-признаки;
    :param data_y: целевая переменная;
    :param model: модель;
    :param metric_path: путь для сохранения метрик.
    """
    result_metrics = create_dict_metrics(
        y_test=data_y,
        y_pred=model.predict(data_x),
        X_test=data_x
    )
    with open(metric_path, 'w') as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла:
    :param config_path: путь до конфигурационного файла;
    :return: метрики.
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['train']['metrics_path']) as json_file:
        metrics = json.load(json_file)

    return metrics
