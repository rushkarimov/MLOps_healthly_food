"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата:
    :param unique_data_path: путь до уникальных значений;
    :param endpoint: endpoint.
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения:
    Section = st.sidebar.selectbox('Section', (unique_df['Section']))
    Category = st.sidebar.selectbox('Category', (unique_df['Category']))
    Type = st.sidebar.selectbox('Type', (unique_df['Type']))
    Manufactured = st.sidebar.selectbox('Manufactured', (unique_df['Manufactured']))
    Price = st.sidebar.slider(
        'Price', min_value=min(unique_df['Price']), max_value=max(unique_df['Price'])
    )


    dict_data = {
        'Section': Section,
        'Category': Category,
        'Type': Type,
        'Manufactured': Manufactured,
        'Price': Price,
    }

    st.write(
        f"""### Выбранные данные:\n
    1) Section: {dict_data['Section']}
    2) Category: {dict_data['Category']}
    3) Typee: {dict_data['Type']}
    4) Manufactured: {dict_data['Manufactured']}
    5) Price: {dict_data['Price']}
    """
    )

    # evaluate and return prediction (text):
    button_ok = st.button('Predict')
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f'## {output}')
        st.success('Success!')


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы:
    :param data: DataFrame;
    :param endpoint: endpoint;
    :param files.
    """
    button_ok = st.button('Predict')
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_['predict'] = output.json()['prediction']
        st.write(data_.head())
