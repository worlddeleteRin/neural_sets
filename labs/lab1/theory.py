import numpy as np
import h5py

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

print('end of importing')


# constants
bit_count = 16
total_count = 2 ** bit_count
learning_sample_size = 10000
# выборка в виде побитового представления
def fit_len(d, count = bit_count):
    if len(d) < count:
        d = [0] * (count - len(d)) + d
    return d

x = np.array([fit_len([int(c) for c in bin(i)[2:]]) for i in range(total_count)])

print('x is', x)
print('x shape is', x.shape)

# массив зависимой переменной
y = np.array([i % 2 for i in range(total_count)])

print('y is', y)
print('y shape is', y.shape)

# разбиваем выборку
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size = 1 - (learning_sample_size / total_count),
    random_state = 42
)

print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)

# Определим классы и приведем переменные к категориальным признакам 
classes = ['четное', 'нечетное']
nb_classes = len(classes)

Y_train = np_utils.to_categorical(y_train, num_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test, num_classes = nb_classes)
Y = np_utils.to_categorical(y, num_classes = nb_classes)

print('Y train is', Y_train)


"""
    Создание сети однослойного персептрона
"""
model = Sequential()
model.add(
    Dense(
        input_dim = bit_count,
        activation = "softmax",
        units = nb_classes,
    )
)

model.summary() # статистика модели
print(model.summary()) # статистика модели


# Компиляция модели

model.compile(
    optimizer = 'adam', # тип градиентного спуска (по умолчанию rmsprop)
    loss = 'categorical_crossentropy', # функция штрафа для классификации
    metrics = ['accuracy'] # вычисляем в модели точность угадываний
)

# Обучение сети

batch_size = 128 # сколько примеров будет скормлено за раз
epochs = 5 # сколько будет прогонов

print(model.fit(
    x_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_data = (x_test, Y_test)
))
"""
loss - функция ошибки
accuracy - точность на обучающей выборке
val_loss - функция ошибка на тестовой выборке
val_accuracy - точность на тестовой выборке
"""

# запись и считывание из файла
model.save('lab1_model_1.h5')

# тестирование модели
# когда считали с файла, нужно проверить результаты работы модели

score = model.evaluate(x_test, Y_test, verbose = 0)
print('test score:', score[0])
print('test accuracy:', score[1])

# увеличим число прогонов
epochs = 10

"""
    Обучаем сеть с новым числом прогонов
"""

print(model.fit(
    x_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_data = (x_test, Y_test)
))
model.save('lab1_model_2.h5')
score = model.evaluate(x_test, Y_test, verbose = 0)
print('test score:', score[0])
print('test accuracy:', score[1])

"""
    Увеличение размера обучающей выборки
"""

learning_sample_size = 20000

print("** Теперь построим и обучим сеть с 20000 ед. выборки **")
x = np.array([fit_len([int(c) for c in bin(i)[2:]]) for i in range(total_count)])
y = np.array([i % 2 for i in range(total_count)])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size = 1 - (learning_sample_size / total_count),
    random_state = 42
)
Y_train = np_utils.to_categorical(y_train, num_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test, num_classes = nb_classes)
Y = np_utils.to_categorical(y, num_classes = nb_classes)

print(model.fit(
    x_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_data = (x_test, Y_test)
))

model.save('lab1_model_3.h5')

"""
    Двухслойный персептрон
"""
model = Sequential()
model.add(
    Dense(
        input_dim = bit_count,
        activation = 'relu',
        units = bit_count // 2
    )
)
model.add(
    Dense(
        activation = 'softmax',
        units = nb_classes
    )
)
print('** Модель с двухслойным персептроном ** \n')
print(model.summary())

# Компиляция модели

model.compile(
    optimizer = 'adam', # тип градиентного спуска (по умолчанию rmsprop)
    loss = 'categorical_crossentropy', # функция штрафа для классификации
    metrics = ['accuracy'] # вычисляем в модели точность угадываний
)

print('** Обучаем модель с двухслойным персептроном ** \n')

print(model.fit(
    x_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 2,
    validation_data = (x_test, Y_test)
))

model.save('lab1_model_4.h5')



