"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import pandas as pd

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import classic_barplot, classic_boxplot, classic_violinplot
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = '../config/params.yml'


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        'https://i.ibb.co/5FM8w5f/Cover-3.png.',
        width=700,
    )

    st.markdown('## Machine learning project:')
    st.markdown('### Предсказания калорийности продуктов 🍏💾📱')
    st.write(
        """
        Многие спортсмены и люди, заботящиеся о своем здоровье, активно следят за своим весом.             
        Ключевым фактором которого является, не количество и длительность тренировок, а количество потребляемых калорий.         
        С помощью данной ML-модели, вы можете предсказать калорийность продукта, заполнив всего 5 пунктов!✅
        """
    )

    # наименования столбцов DataFrame
    st.markdown(
        """
        ### Описание полей: 
            - Section - раздел;
            - Category - категория; 
            - Type - тип;
            - Manufactured - производитель;
            - Price - цена;
            - Energy_value - калорийность(target).
        """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown('# Exploratory data analysis️')

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # загрузка и чтение train DataFrame
    data = get_dataset(dataset_path=config['preprocessing']['train_path'])
    st.write(data.head())

    # Отрисовка графиков с помощью checkbox
    All_Section_Energy_value = st.sidebar.checkbox('В каком разделе большая калорийность?')
    Chocolate_Manufactured_Energy_value = st.sidebar.checkbox('Дешевый шоколад от Глобуса или Alpen Gold?')
    Bar_Manufactured_Energy_value = st.sidebar.checkbox('Спортивный батончик Bombbar или классический Mars?')
    Price_Energy_value = st.sidebar.checkbox('Йогурт, мороженое или печенье?')
    Brand_Manufactured_Energy_value = st.sidebar.checkbox('Зная бренд, можно ли узнать калорийность?')
    Child_Type_Energy_value = st.sidebar.checkbox('Продукты для детей более калорийные, чем для взрослых?')
    Diabet_Section_Energy_value = st.sidebar.checkbox('В сладостях для диабетиков меньше сахара?')


    if All_Section_Energy_value:
        st.markdown(
            """
            ##### 1 гипотеза:
            ##### В разделах  кондитерские изделия и сладости большая калорийность, 
            ##### а в разделах овощи, фрукты, зелень маленькая?
            """
        )

        st.pyplot(
            classic_barplot(
                data, 'Energy_value', 'Section', 'Section - Energy_value'
            )
        )

        st.pyplot(
            classic_boxplot(
                data, 'Energy_value', 'Section', 'Section - Energy_value'
            )
        )

        st.markdown(
            """
            В разделах сладости, кондитерские изделия, а также бакалея большая калорийность, в среднем 
            больше 300 калорий.                                                                                                         
            А в разделах овощи, фрукты, зелень меньше 50 калорий.
            """
        )


    if Chocolate_Manufactured_Energy_value:
        st.markdown(
            """
            ##### 2 гипотеза:
            ##### Продукты дешевого сегмента, например, шоколад, произведенный под 
            ##### брендом самого гипермаркета(Ашана, Глобуса и т.д.), более калорийный,
            ##### чем шоколад среднего сегмента и бренда, например, Alpen Gold?
            """
        )

        # фильтр по разделу и категории шоколад
        df_chocolate = data[(data['Section'] == 'Хлеб, кондитерские изделия')
                          & (data['Category'] == 'Шоколад, конфеты, жевательная резинка')
                          & (data['Type'] == 'Шоколад и шоколадные изделия')]

        # фильтр по брендам
        df_chocolate_two_brand = df_chocolate[
            (df_chocolate['Manufactured'] == 'ООО "Кондитерская фабрика "Волшебница"')
            | (df_chocolate['Manufactured'] == '''ООО "Мон'дэлис Русь"''')]

        rename_chocolate = {
            '''ООО "Мон'дэлис Русь"''': 'Alpen Gold',
            '''ООО "Кондитерская фабрика "Волшебница"''': 'Globus'
        }

        df_chocolate_two_brand['Manufactured'] = df_chocolate_two_brand[
            'Manufactured'].replace(rename_chocolate)

        st.pyplot(
            classic_boxplot(
                df_chocolate_two_brand, 'Manufactured', 'Energy_value',
                'Manufactured - Energy_value', 'dark:salmon_r'
            )
        )

        st.markdown(
            """
            Да, у Alpen Gold в среднем 510 калорий, а у Глобуса 550 калорий.
            """
        )


    if Bar_Manufactured_Energy_value:
        st.markdown(
            """
            ##### 3 гипотеза:
            ##### Продукты для спортсменов, например, протеиновые батончиники  
            ##### Bombbar или Ironman менее калорийные, чем классические батончики 
            ##### Kit-Kat, Nuts (Nestle) или Snickers, Twix (Mars)?
            """
        )

        # фильтр по разделу и категории спортивные батончики
        df_protein_bar = data[(data['Section'] == 'Бакалея')
                            & (data['Category'] == 'Диетические продукты') &
                            (data['Type'] == 'Спортивное питание')]

        # фильтр по брендам спортивные батончики
        df_protein_bar_two_brand = df_protein_bar[
            (df_protein_bar['Manufactured'] ==
             'ООО "АРТ Современные научные технологии"') |
            (df_protein_bar['Manufactured'] == '''ООО "Фитнес Фуд"''')]

        # фильтр по разделу и категории классические батончики
        df_chocolate_bar = data[(data['Section'] == 'Хлеб, кондитерские изделия') & (
                data['Category'] == 'Шоколад, конфеты, жевательная резинка') &
                              (data['Type'] == 'Шоколад и шоколадные изделия')]

        # фильтр по брендам классические батончики
        df_chocolate_bar_two_brand = df_chocolate_bar[
            (df_chocolate_bar['Manufactured'] == 'ООО "Марс"') |
            (df_chocolate_bar['Manufactured'] == '''ООО "Нестле Россия"''')]

        # соединение в единый DataFrame
        df_bar = pd.concat([
            df_protein_bar_two_brand[['Manufactured', 'Energy_value']],
            df_chocolate_bar_two_brand[['Manufactured', 'Energy_value']]
        ])

        rename_bar = {
            '''ООО "АРТ Современные научные технологии"''': 'Ironman',
            '''ООО "Фитнес Фуд"''': 'Bombbar',
            '''ООО "Марс"''': 'Mars',
            '''ООО "Нестле Россия"''': 'Nestle'
        }

        df_bar['Manufactured'] = df_bar['Manufactured'].replace(rename_bar)

        st.pyplot(
            classic_boxplot(
                df_bar, 'Energy_value', 'Manufactured',
                'Manufactured - Energy_value', 'dark:salmon_r'
            )
        )

        st.markdown(
            """
            Да, в спортивных батончиках в среднем до 300 калорий, а в классических около 500.
            """
        )


    if Price_Energy_value:
        st.markdown(
            """
            ##### 4 гипотеза:
            ##### Дорогие продукты имеют большую калорийность, чем дешевые, 
            ##### например, в категориях: йогурты, мороженое и печенье?
            """
        )

        st.markdown(
            """
            ###### Йогурты:
            """
        )

        # фильтр по разделу и категории традиционных йогуртов
        df_yogurt = data[(data['Section'] == 'Молочные продукты, сыр, яйца')
                        & (data['Category'] == 'Йогурт') &
                        (data['Type'] == 'Йогурты традиционные')]

        st.pyplot(
            classic_barplot(
                df_yogurt, 'Price', 'Energy_value',
                'Yogurt(Price - Energy_value)', 'Blues', 90
            )
        )

        st.pyplot(
            classic_boxplot(
                df_yogurt, 'Price', 'Energy_value',
                'Yogurt(Price - Energy_value)', 'Blues', 90
            )
        )

        st.markdown(
            """
            ###### Мороженое:
            """
        )

        # фильтр по разделу и категории мороженое
        df_ice_cream = data[(data['Section'] == 'Замороженные продукты')
                          & (data['Category'] == 'Мороженое, десерты')]

        st.pyplot(
            classic_barplot(
                df_ice_cream, 'Price', 'Energy_value',
                'Ice cream(Price - Energy_value)', 'light:b',
                90, 11, 11
            )
        )

        st.pyplot(
            classic_boxplot(
                df_ice_cream, 'Price', 'Energy_value',
                'Ice cream(Price - Energy_value)', 'light:b',
                90, 11, 11
            )
        )

        st.markdown(
            """
            ###### Печенья:
            """
        )

        # фильтр по разделу и категории печенья
        df_cookies = data[(data['Section'] == 'Хлеб, кондитерские изделия')
                        & (data['Category'] == 'Печенье, пряники, вафли') &
                        (data['Type'] == 'Печенье, галеты, крекеры')]

        st.pyplot(
            classic_barplot(
                df_cookies, 'Price', 'Energy_value',
                'Сookies(Price - Energy_value)', 'YlOrBr',
                90, 8, 8
            )
        )

        st.pyplot(
            classic_boxplot(
                df_cookies, 'Price', 'Energy_value',
                'Сookies(Price - Energy_value)', 'YlOrBr',
                90, 8, 8
            )
        )

        st.markdown(
            """
            В данных категориях взаимосвязь не обнаружена, но это не исключает тот факт, что она может 
            присутствовать в других категориях.
            """
        )


    if Brand_Manufactured_Energy_value:
        st.markdown(
            """
            ##### 5 гипотеза:
            ##### Как правило, производители специализируются на определенных 
            ##### категориях продуктов, и зная производителя, можно предположить, 
            ##### какая в среднем будет калорийность его продуктов?
            """
        )

        st.markdown(
            """
            ###### Самые популярные производители:
            """
        )

        # вывод наиболее популярных производителей
        df_Manufactured = data.Manufactured.value_counts(
            normalize=True)[:10].to_frame().rename(columns={
            'proportion': 'Percent'
        }).reset_index()

        st.pyplot(
            classic_barplot(
                df_Manufactured, 'Percent', 'Manufactured',
                'Manufactured - Percent', 'icefire'
            )
        )

        st.markdown(
            """
            ###### Взаимосвязь производителя и калорийности:
            """
        )

        # фильтр по топ 5-ти производителям
        df_Globus = data[(data['Manufactured'] == 'ООО "Гиперглобус"') |
                       (data['Manufactured'] == 'ООО "Марс"') |
                       (data['Manufactured'] == 'АО "Данон Россия"') |
                       (data['Manufactured'] == 'АО "Прогресс"') |
                       (data['Manufactured'] == 'ООО "Нестле Россия"')]

        st.pyplot(
            classic_boxplot(
                df_Globus, 'Manufactured', 'Energy_value',
                'Manufactured - Energy_value', 'icefire'
            )
        )

        st.markdown(
            """
            Калорийность зависит от производителя и категорий, например, "Глобус", "Марс" и "Нестле" 
            выпускают продукцию в разных категориях, поэтому взаимосвязи не наблюдается, при этом "Даннон"
            специализируется на йогуртах, поэтому у них в среднем около 100 калорий, "Прогресс" на детском
            питании, поэтому около 80 калорий.
            """
        )


    if Child_Type_Energy_value:
        st.markdown(
            """
            ##### 6 гипотеза:
            ##### Продукты для детей более калорийные, чем для взрослых?
            """
        )

        # фильтр по разделу и категории детские каши
        df_porridge_children = data[(data['Section'] == 'Детские товары')
                                  & (data['Category'] == 'Детское питание') &
                                  (data['Type'] == 'Детские каши')]

        # фильтр по разделу и категории взрослые каши
        df_porridge_adults = data[(data['Section'] == 'Бакалея')
                                & (data['Category'] == 'Продукты быстрого приготовления')
                                & (data['Type'] == 'Хлопья и каши')]

        # соединение в единый DataFrame
        df_porridge = pd.concat([
            df_porridge_children[['Type', 'Energy_value']],
            df_porridge_adults[['Type', 'Energy_value']]
        ])

        rename_porridge = {'Хлопья и каши': 'Классические каши'}

        df_porridge['Type'] = df_porridge['Type'].replace(rename_porridge)

        st.pyplot(
            classic_violinplot(
                df_porridge, 'Energy_value', 'Type', 'Type - Energy_value',
                'cubehelix'
            )
        )

        st.markdown(
            """
            Да, небольшая разница на 20-30 калорий присутствует.
            """
        )


    if Diabet_Section_Energy_value:
        st.markdown(
            """
            ##### 7 гипотеза:
            ##### В сладостях для диабетиков используются сахорозаменители, поэтому они 
            ##### менее калорийные, чем обычные сладости?
            """
        )

        # фильтр по категории для диабетиков
        df_sweets_diabetics = data[(data['Section'] == 'Бакалея')
                                 & (data['Category'] == 'Диетические продукты') &
                                 (data['Type'] == 'Заменители сахара и диабетика')]

        rename_diabetics = {'Бакалея': 'Диабетические'}

        df_sweets_diabetics['Section'] = df_sweets_diabetics['Section'].replace(
            rename_diabetics)

        # фильтр по категории кондитерские изделия
        df_sweets_classic = data[(data['Section'] == 'Хлеб, кондитерские изделия')]

        rename_classic = {'Хлеб, кондитерские изделия': 'Классические'}

        df_sweets_classic['Section'] = df_sweets_classic['Section'].replace(
            rename_classic)

        # соединение в единый DataFrame
        df_sweets = pd.concat([
            df_sweets_diabetics[['Section', 'Energy_value']],
            df_sweets_classic[['Section', 'Energy_value']]
        ])

        st.pyplot(
            classic_boxplot(
                df_sweets, 'Section', 'Energy_value', 'Section - Energy_value',
                'viridis'
            )
        )

        st.markdown(
            """
            Да, есть разница где-то на 50 калорий.
            """
        )

def training():
    """
    Тренировка модели
    """
    st.markdown('# Training model LightGBM')
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config['endpoints']['train']

    if st.button('Start training'):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown('# Prediction')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_input']
    unique_data_path = config['preprocessing']['unique_values_path']

    # проверка на наличие сохраненной модели
    if os.path.exists(config['train']['model_path']):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error('Сначала обучите модель')


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown('# Prediction')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_from_file']

    upload_file = st.file_uploader(
        '', type=['csv', 'xlsx'], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data='Test')
        # проверка на наличие сохраненной модели
        if os.path.exists(config['train']['model_path']):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error('Сначала обучите модель')


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        'Описание проекта📄': main_page,
        'Анализ данных📊': exploratory,
        'Тренировка ML-модели🏃‍♂️️': training,
        'Узнай калорийность продукта🍏': prediction,
        'Узнай, загрузив свой файл💻': prediction_from_file,
    }
    selected_page = st.sidebar.selectbox('Выберите пункт', page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == '__main__':
    main()
