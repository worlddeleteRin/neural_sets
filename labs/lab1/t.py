import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

print('end of importing')


# constants
bit_count = 16
total_count = 2 ** bit_count
learning_sample_size = 10000

# выборка в виде побитового предоставления
def fit_len(d, count = bit_count):
    if len(d) < count:
        d = [0] * (count - len(d)) + d
    return d

x = np.array([fit_len([int(c) for c in bin(i)[2:]]) for i in range(total_count)])

print('x is', x)

# массив зависимой переменной
y = np.array([i % 2 for i in range(total_count)])

print('y is', y)

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

model.fit(
    x_train,
    Y_train,
    batch_size = 128,
    epochs = 5,
    verbose = 2,
    validation_data = (x_test, Y_test)
)



