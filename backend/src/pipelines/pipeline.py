"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import os
import joblib
import yaml

from ..data.split_dataset import split_train_test
from ..train.train import find_optimal_params, train_model
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели:
    :param config_path: путь до файла с конфигурациями;
    :return: None.
    """
    # получение данных
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['train']

    # получение данных
    train_data = get_dataset(dataset_path=preprocessing_config['train_path'])

    # предобработка
    train_data = pipeline_preprocess(data=train_data, flg_evaluate=False, **preprocessing_config)

    # разделение данных
    df_train, df_test = split_train_test(dataset=train_data, **preprocessing_config)

    # поиск оптимальных параметров
    study = find_optimal_params(data_train=df_train, data_test=df_test, **train_config)

    # тренировка с оптимальными параметрами
    clf = train_model(
        data_train=df_train,
        data_test=df_test,
        study=study,
        target=preprocessing_config['target_column'],
        metric_path=train_config['metrics_path'],
        test_size=preprocessing_config['test_size'],
        random_state=preprocessing_config['random_state']
    )

    # сохранение результатов (study, model)
    joblib.dump(clf, os.path.join(train_config['model_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))
