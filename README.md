## UI Demo
![alt text](demo/example.gif?raw=true)

# Инструкция

### Запуск FastAPI:

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload`

Ссылка:         
http://localhost:8000/docs

В случае ошибки, убить все процессы:

`for pid in $(ps -ef | grep "uvicorn main:app" | awk '{print $2}'); do kill -9 $pid; done`
___

### Запуск Streamlit:

`cd frontend`

`streamlit run main.py`

Ссылка:           
http://localhost:8501 

В случае ошибки, убить все процессы:

`for pid in $(ps -ef | grep "streamlit run" | awk '{print $2}'); do kill -9 $pid; done`

___
### Folders
- `/backend` - Папка с проектом FastAPI;
- `/frontend` - Папка с проектом Streamlit;
- `/config` - Папка, содержащая конфигурационный файл;
- `/data` - Папка, содержащая исходные данные, обработанные данные, уникальные значения в формате JSON, а также неразмеченный файл для подачи на вход модели;
- `/demo` - Папка, содержащая демо работы сервиса в Streamlit UI в формате gif;
- `/models` - Папка, содержащая сохраненную модель после тренировки, а также объект study (Optuna);
- `/notebooks` - Папка, содержащая jupyter ноутбуки с предварительным анализом данных;
- `/report` - Папка, содержащая информацию о лучших параметрах и метриках после обучения.
