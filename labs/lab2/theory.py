import math
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import List

print('eof imports')

# constants 
interval_start = -2 # начало интервала по x для выборки
interval_end = 4 # конец интервала по x для выборки
total_interval_count = 1000 # всего интервалов
learning_sample_size = 200 # кол-во точек в выборке

epoch_count = 20 # максимальное кол-во эпох
neiron_count = 20 # кол-во нейронов в скрытом слое

"""
    Создаем выборку. Строим интервалы по входной переменной
"""
print('** Создаем выборку, строим интервалы для входной переменной **')

# разбиваем отрезок interval_start, interval_end на интервалы с равномерным шагом

x = np.array(
    [
        interval_start + (interval_end - interval_start) * i / total_interval_count
        for i in range(total_interval_count + 1)
    ]
)

print('Интервал: \n', x)

"""
    Определяем функциональную связь, которую будем восстанавливать
"""
print('** Определяем функциональную связь, которую будем восстанавливать **')

def relation_spot(x):
    return 0.5 * (x - 1) ** 2 - 1
    
# рассчитываем значение функции для интервалов
y = np.array(
    [relation_spot(i) for i in x] 
)
print('Значение функции для интервалов: \n', y)

"""
    Построение графика исходной функциональной зависимости
"""
print('** Построение графика исходной функциональной зависимости **')

plt.plot(x, y, label = 'функция')
ax = plt.gca()
ax.axhline(y = 0, color = 'k') # рисуем ось X
ax.axvline(x = 0, color = 'k') # рисуем ось Y
plt.legend() # добавляем легенду
plt.show()

"""
    Разбиение выборки на обучающую и тестовую
"""
print('** Разбиение выборки на обучающую и тестовую **')

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 1.0 - learning_sample_size / total_interval_count
)
print(f'x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
