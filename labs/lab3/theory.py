"""
Лабораторная работа 3 
Теоретическая часть, шаблон лабораторной
"""

import yfinance as yf

import numpy as np
import pandas as pd

import h5py
import time
import os
import shutil

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.models import load_model

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
"""
    import kerastuner is deprecated.
    Need to use - import keras_tuner
"""
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization

timer_start = time.time()
# constants
best_model_name = "best_model.h5"
save_plots_folder = "plots"
is_model_ready = False
save_plots = True
max_trials = 100 # максимальное кол-во запусков обучения

# clean dirs
try:
    shutil.rmtree('lab3_dir', ignore_errors = True)
    print('dirs cleaned!')
except Exception as e:
    print('exception while rm dirs: ', e)
    pass

"""
    Загрузка исходных данных
"""
print('** Загрузка исходных данных **')

y_name = 'IBM'
date_start = '2016-01-01'
date_end = '2019-08-01'

data = pd.DataFrame(columns = [y_name])
data[y_name] = yf.download([y_name], date_start, date_end)['Adj Close']

print(data.head())

"""
    Построение графика исходного временного ряда
"""
# plot all close prices
((data.pct_change()+1).cumprod()).plot(figsize = (10,7))
plt.legend()
plt.title(
    "Adjusted Close Price",
    fontsize = 16
)
plt.ylabel('Price', fontsize = 14)
plt.xlabel('Year', fontsize = 14)
plt.grid(
    which = "major",
    color = "k",
    linestyle = "-",
    linewidth = 0.5
)

"""
    Исходные данные для авторегрессионной зависимости
"""
net_data = data.copy()
d = data[y_name]

for shift in range(1,6):
    net_data[y_name + '_' + str(shift)] = d.shift(shift)

net_data = net_data[5:]
print('net data is', net_data)

y = net_data.pop(y_name)

train_size = int(len(y) * 0.7)

x_train = net_data[0:train_size]
y_train = y[0:train_size]
x_test = net_data[train_size:]
y_test = y[train_size:]

print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')


"""
    Создаем нейронную сеть
"""
print('Создаем нейронную сеть')

model = Sequential()
model.add(
    Dense(
        50,
        activation = 'relu',
        input_shape = (5,)
    )
)
model.add(
    Dense(
        50,
        activation = 'relu',
    )
)
model.add(
    Dense(1)
)
model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mae']
)

model.summary()

"""
    Обучение сети
"""

history = model.fit(
    x_train, y_train,
    batch_size = 256,
    epochs = 20,
    verbose = 2,
    validation_data = (x_test, y_test)
)

"""
    Оценка точности работы сети
"""
plt.figure(2)
plt.plot(
    history.history['mae'], 
    label = 'Обучающая выборка',
)
plt.plot(
    history.history['val_mae'], 
    label = 'Тестовая выборка',
)
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()

mse, mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Среднеквадратичное отклонение: {mse}')
print(f'Средняя абсолютная ошибка: {mae}')


"""
    Визуализация работы сети
"""
y_pred = model.predict(net_data)
data_predicted = data[5:].copy()
data_predicted[y_name + '_predicted'] = y_pred

# Строим для обучающей выборки
# plt figure 3
((data_predicted[:train_size].pct_change() +1).cumprod()).plot(figsize = (10,7))
plt.legend()
plt.title("Train", fontsize = 16)

plt.ylabel('Price', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

plt.grid(
    which = "major",
    color = 'k',
    linestyle = '-',
    linewidth = 0.5
)

# Строим для тестовой выборки
# plt figure 4
((data_predicted[train_size:].pct_change() +1).cumprod()).plot(figsize = (10,7))
plt.legend()
plt.title("Test", fontsize = 16)

plt.ylabel('Price', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

plt.grid(
    which = "major",
    color = 'k',
    linestyle = '-',
    linewidth = 0.5
)

"""
    Оптимизация гиперпараметров нейронной сети
"""

def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice(
        'activation',
        values=['relu', 'sigmoid', 'tanh', 'elu', 'selu']
    )
    model.add(
        Dense(
            # полносвязный слой с разным кол-вом нейронов
            units = hp.Int(
                'units_input', # min кол-во нейронов
                min_value = 4,
                max_value = 256, # max кол-во нейронов
                step = 4
            ),
            input_dim = 5,
            activation = activation_choice
        )
    )
    model.add(
        Dense(
            units = hp.Int(
                'units_hidden',
                min_value = 4,
                max_value = 256,
                step = 4
            ),
            activation = activation_choice
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer = hp.Choice(
            'optimizer',
            values = ['adam', 'rmsprop', 'SGD']
        ),
        loss = 'mse',
        metrics = ['mae']
    )
    return model

"""
    Подбор гиперпараметров
"""
# создадим тюнер с помощью типа RandomSearch
tuner = RandomSearch(
    build_model, # функция создания модели
    objective = 'val_mae', # метрика, которую нужно оптимизировать
    max_trials = max_trials, # максимальное кол-во запусков обучения
    directory = 'lab3_dir' # куда будут сохраняться обученные сети
) 

tuner.search_space_summary() # выводим статистику

if not is_model_ready:
    print('** Начинается большое обучение через 3 сек., готовимся ждать... **')
    tuner.search(
        x_train, # Данные для обучения
        y_train, # Правильные ответы
        batch_size = 256, # Размер мини-выборки
        epochs = 20, # кол-во эпох обучения
        validation_data = (x_test, y_test) # Данные для проверки
    )
    """
    В результате будет произведен расчет 100 моделей с различной архитектурой, 
    результаты прогнозирования сравнены и ранжированы.
    """

    """
        Выбираем лучшую модель
    """

    print('** Выбираем лучшую модель **')
    models = tuner.get_best_models(num_models = 1)
    model = models[0]
    # сохраняем модель
    model.save(best_model_name)
else:
    model = load_model(best_model_name)

print(model.summary())
print(model.evaluate(x_test, y_test))

# рассчитаем прогнозные значения
y_pred = model.predict(net_data)
data_predicted[y_name + '_predicted'] = y_pred

# построим графики
((data_predicted[:train_size].pct_change() +1).cumprod()).plot(figsize = (10,7))
plt.legend()
plt.title("Train", fontsize = 16)

plt.ylabel('Price', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

plt.grid(
    which = "major",
    color = 'k',
    linestyle = '-',
    linewidth = 0.5
)

# Строим для тестовой выборки
# plt figure 4
((data_predicted[train_size:].pct_change() +1).cumprod()).plot(figsize = (10,7))
plt.legend()
plt.title("Test", fontsize = 16)

plt.ylabel('Price', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

plt.grid(
    which = "major",
    color = 'k',
    linestyle = '-',
    linewidth = 0.5
)

mse, mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Среднеквадратичное отклонение: {mse}')
print(f'Средняя абсолютная ошибка: {mae}')

timer_end = time.time()


if save_plots:
    print('Сохраняем графики...')
    try:
        try:
            shutil.rmtree(save_plots_folder, ignore_errors = True)
        except Exception as e:
            print(f'exception при удалении папки графиков: {e}')
        os.mkdir(save_plots_folder)
    except Exception as e:
        print('Не удалось сохранить графики', e)
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for index, figure in enumerate(figures):
        figure.savefig(
            f'{save_plots_folder}/plot-{index}.png', 
            dpi = 300, 
            bbox_inches='tight'
        )

print(f'Скипт исполнялся : {timer_end - timer_start} секунд')

# plt.show()

