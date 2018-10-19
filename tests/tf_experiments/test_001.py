##################################################################
## http://wiki.programstore.ru/python-tensorflow-dlya-chajnikov-chast-1/
#
# Простейший пример использования обучения tensorflow-графа.
# Граф: константа (=1) умноженная на переменную (нач.знач. =0.8)
# Требуемое значение на выходе: 2
# Количество итераций: 100

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

# результат:
#   1.9928955   -- результирующее значение после обучения
#   + график, демонстрирующимй уменьшение ошибки
