"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов:
    :param config: конфигурационный файл;
    :param endpoint: endpoint.
    """
    # Last metrics
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'R2_adjusted': 0, 'WAPE': 0}

    # Train
    with st.spinner('Модель подбирает параметры...'):
        output = requests.post(endpoint, timeout=8000)
    st.success('Success!')

    new_metrics = output.json()['metrics']

    # diff metrics
    MAE, MSE, RMSE, R2_adjusted, WAPE = st.columns(5)
    MAE.metric(
        'MAE',
        new_metrics['MAE'],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",
    )
    MSE.metric(
        'MSE',
        new_metrics['MSE'],
        f"{new_metrics['MSE']-old_metrics['MSE']:.3f}",
    )
    RMSE.metric(
        'RMSE',
        new_metrics['RMSE'],
        f"{new_metrics['RMSE']-old_metrics['RMSE']:.3f}",
    )
    R2_adjusted.metric(
        'R2_adjusted',
        new_metrics['R2_adjusted'],
        f"{new_metrics['R2_adjusted']-old_metrics['R2_adjusted']:.3f}"
    )
    WAPE.metric(
        'WAPE',
        new_metrics['WAPE'],
        f"{new_metrics['WAPE']-old_metrics['WAPE']:.3f}",
    )

    # plot study
    study = joblib.load(os.path.join(config['train']['study_path']))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
