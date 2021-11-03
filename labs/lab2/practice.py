
"""
    Лабораторная работа 2
    Вариант 14
    Функциональная связь -  ln(2x^2 - 3x + 2)
    Интервал построения - [1;4]
    Стохастическая компонента - мультипликативная, равномерное на [1,2;1,8]
"""

import math
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import List
import random

print('eof imports')

# constants 
interval_start = 1 # начало интервала по x для выборки
interval_end = 4 # конец интервала по x для выборки
total_interval_count = 1000 # всего интервалов
learning_sample_size = 200 # кол-во точек в выборке

epoch_count = 200 # максимальное кол-во эпох
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
    return np.log((2 * x) ** 2 - 3 * x + 2)
    
# рассчитываем значение функции для интервалов
y = np.array(
    [relation_spot(i) for i in x] 
)
print('Значение функции для интервалов: \n', y)

"""
    Построение графика исходной функциональной зависимости
"""
print('** Построение графика исходной функциональной зависимости **')

plt.figure(0)
plt.plot(x, y, label = 'функция')
ax = plt.gca()
ax.axhline(y = 0, color = 'k') # рисуем ось X
ax.axvline(x = 0, color = 'k') # рисуем ось Y
plt.legend() # добавляем легенду
# plt.show()

"""
    Разбиение выборки на обучающую и тестовую
"""
print('** Разбиение выборки на обучающую и тестовую **')

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 1.0 - learning_sample_size / total_interval_count
)
print(f'x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')

"""
    Визуализация обучающей выборки
"""
print('Визуализация обучающей выборки')

# сортировка выборки по переменной x
z = zip(x_train, y_train)
# первый элемент каждого списка используем как ключ
z_sorted = sorted(z, key = lambda tup: tup[0])
# извлекаем x и y
x1 = [z[0] for z in z_sorted]
y1 = [z[1] for z in z_sorted]

print('** Строим график на основании исходной и обучающей выборки **')

plt.figure(1)
plt.plot(
    x, y,
    label = 'Исходные'
)
plt.plot(
    x1, y1,
    label = 'Обучающие'
)
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()
# plt.show()

"""
    Нормализация данных
"""
x_mean = x_train.mean(axis = 0) # Среднее значение 
x_std = x_train.std(axis = 0) # Стандартное отклонение

print(f'Среднее значение : {x_mean}')
print(f'Стандартное отклонение : {x_std}')

# нормируем входные данные для обучающей и тестовой выборки
x_train -= x_mean
x_train /= x_std
x_test -= x_mean
x_test /= x_std

print(f'mean value after normalization: {x_train.mean(axis = 0)}')
print(f'std deviation after normalization: {x_train.std(axis = 0)}')

"""
    Создаем нейронную сеть
"""
print('Создаем нейронную сеть')

model = Sequential()
model.add(
    Dense(
       neiron_count,
       activation = 'relu',
       input_shape = (1,)
    )
)
model.add(
    Dense(
        neiron_count,
        activation = 'relu'
    )
)
model.add(Dense(1))

model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mae']
)

print(model.summary())
# loss mse - функция ошибки среднеквадратичное отклонение
# metics mae - метрика - среднее абсолютное отклонение

"""
    Обучение сети
"""
print('** Обучение сети **')

history = model.fit(
    x_train, y_train,
    batch_size = 20,
    epochs = epoch_count,
    verbose = 2,
    validation_data = (x_test, y_test)
)

"""
    Оценка точности работы сети
"""
print('Оценка точности работы сети')

print(history.history.keys())

# построим график результирующих параметров процесса обучения сети

plt.figure(2)
plt.plot(
    history.history['mae'],
    label = 'Обучающая выборка'
)
plt.plot(
    history.history['val_mae'], 
    label = 'Тестовая выборка'
)
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
# plt.show()

# проверка точности нейронной сети на тестовой выборке
mse, mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Среднеквадратичное отклонение : {mse}')
print(f'Средняя абсолютная ошибка: {mae}')

"""
    Визуализация работы сети
"""
# нормируем полную выборку
x_norm = x - x_mean
x_norm /= x_std

# рассчитваем выходные значения по полной выборке
y_pred = model.predict(x_norm)

print(f'Среднее значение после визуализации: {x_norm.mean(axis = 0)}')
print(f'Стандартное отклонение после нормализации: {x_norm.std(axis = 0)}')

# сравниваем реальные и прогнозные значения
plt.figure(3)
plt.plot(x, y, label = 'Исходные')
plt.plot(x, y_pred, label = 'Прогнозные')
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()
# plt.show()


"""
    Оценка влияния стохастической компоненты на обучаемость
    нейронной сети
"""

train_start_int = 1.2
train_stop_int = 1.8

def variate(z, start_interval, stop_interval):
    return z * random.uniform(start_interval, stop_interval)


# снова разбиваем данные
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 1.0 - learning_sample_size / total_interval_count
)

# применяем влияение стох компоненты к обучаяющим данным
y_train = np.array(
    [
        variate(i, train_start_int, train_stop_int) for i in y_train
    ]
)

# сортируем для отображения
z = zip(x_train, y_train)
z_sorted = sorted(z, key = lambda tup: tup[0])

x1 = [z[0] for z in z_sorted]
y1 = [z[1] for z in z_sorted]

plt.figure(4)
plt.plot(x, y, label = 'Исходные')
plt.plot(x1, y1, label = 'Обучающие')
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()


# test_m = 1
# test_sko = 0.1
# применяем влияение стох компоненты к тестовым данным
y_test = np.array(
    [
        variate(i, train_start_int, train_stop_int) for i in y_test
    ]
)

# сортируем для отображения
z = zip(x_test, y_test)
z_sorted = sorted(z, key = lambda tup: tup[0])
x1 = [z[0] for z in z_sorted]
y1 = [z[1] for z in z_sorted]

plt.figure(5)
plt.plot(x, y, label = 'Исходные')
plt.plot(x1, y1, label = 'Тестовая')
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()

"""
    Нормализация данных
"""
x_mean = x_train.mean(axis = 0) # Среднее значение 
x_std = x_train.std(axis = 0) # Стандартное отклонение

print(f'Среднее значение : {x_mean}')
print(f'Стандартное отклонение : {x_std}')

# нормируем входные данные для обучающей и тестовой выборки
x_train -= x_mean
x_train /= x_std
x_test -= x_mean
x_test /= x_std

print(f'mean value after normalization: {x_train.mean(axis = 0)}')
print(f'std deviation after normalization: {x_train.std(axis = 0)}')

"""
    Обучение сети
"""
print('** Обучение сети **')

history = model.fit(
    x_train, y_train,
    batch_size = 20,
    epochs = epoch_count,
    verbose = 2,
    validation_data = (x_test, y_test)
)

"""
    Оценка точности работы сети
"""
plt.figure(6)
plt.plot(
    history.history['mae'],
    label = 'Обучающая выборка'
)
plt.plot(
    history.history['val_mae'], 
    label = 'Тестовая выборка'
)
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
# plt.show()

# проверка точности нейронной сети на тестовой выборке
mse, mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Среднеквадратичное отклонение : {mse}')
print(f'Средняя абсолютная ошибка: {mae}')

"""
    Визуализация работы сети
"""
# нормируем полную выборку
x_norm = x - x_mean
x_norm /= x_std

# рассчитваем выходные значения по полной выборке
y_pred = model.predict(x_norm)

print(f'Среднее значение после нормализации: {x_norm.mean(axis = 0)}')
print(f'Стандартное отклонение после нормализации: {x_norm.std(axis = 0)}')

# сравниваем реальные и прогнозные значения
plt.figure(7)
plt.plot(x, y, label = 'Исходные')
plt.plot(x, y_pred, label = 'Прогнозные')
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()


"""
    Использование EarlyStopping Callback для остановки обучения нейросети 
    при переобучении
"""

early_stopping_callback = EarlyStopping(monitor = 'val_mae', patience = 3)

history = model.fit(
    x_train, y_train,
    batch_size = 20,
    epochs = epoch_count,
    verbose = 2,
    validation_data = (x_test, y_test),
    callbacks = [early_stopping_callback]
)
print('Обучение остановлено на эпохе', early_stopping_callback.stopped_epoch)

"""
    Оценка точности работы сети
"""
plt.figure(8)
plt.plot(
    history.history['mae'],
    label = 'Обучающая выборка'
)
plt.plot(
    history.history['val_mae'], 
    label = 'Тестовая выборка'
)
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
# plt.show()

# проверка точности нейронной сети на тестовой выборке
mse, mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Среднеквадратичное отклонение : {mse}')
print(f'Средняя абсолютная ошибка: {mae}')

"""
    Визуализация работы сети
"""
# нормируем полную выборку
x_norm = x - x_mean
x_norm /= x_std

# рассчитваем выходные значения по полной выборке
y_pred = model.predict(x_norm)

# сравниваем реальные и прогнозные значения
plt.figure(9)
plt.plot(x, y, label = 'Исходные')
plt.plot(x, y_pred, label = 'Прогнозные')
ax = plt.gca()
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.legend()

plt.show()
