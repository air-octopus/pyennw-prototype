# import numpy as np
#
# import tensorflow as tf
#
# in_01 = tf.constant(tf.zeros([2]))
# v_01 = tf.Variable(1)
# #v_02 = v_01.
# out_01 = v_01 * in_01[1]
#
# sess = tf.Session()
# print(sess.run(out_01))

##################################################################
## http://wiki.programstore.ru/python-tensorflow-dlya-chajnikov-chast-1/

import tensorflow as tf
import matplotlib.pyplot as plt

# Создадим граф
graph = tf.get_default_graph()

# Создадим входное значение
input_value = tf.constant(1.0)

# Создадим переменную
weight = tf.Variable(0.8)

# Создадим выходное значение
output_value = weight * input_value

# создадим сессию
sess = tf.Session()

# Создаем оптимизиатор
desired_output_value = tf.constant(2.0)
loss = (output_value - desired_output_value) ** 2  # Функция ошибки
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)  # Оптимизатор
grads_and_vars = optim.compute_gradients(loss)

# Инициализируем переменные
init = tf.global_variables_initializer()
sess.run(init)

# задаем начальные массивы
x = []
y = []

# Обучаем
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
train_step = optim.minimize(loss)
for i in range(100):
    sess.run(train_step)
    x.append(i)
    y.append(sess.run(loss))

print(sess.run(output_value))

# Строим график
fig = plt.figure()
plt.plot(x, y)

# Отображаем заголовки и подписи осей
plt.title('График функции')
plt.ylabel('Ось Y')
plt.xlabel('Ось X')
plt.grid(True)
plt.show()
