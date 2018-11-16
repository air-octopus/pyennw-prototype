##################################################################
## Модифицированный вариант примера из test_001.py для иллюстрации использования tf.summary и tensorboard
#

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

tf.summary.scalar('loss', loss)
merged_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/home/maksko/-/Temp/001', graph)

optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)  # Оптимизатор
grads_and_vars = optim.compute_gradients(loss)

# Инициализируем переменные
init = tf.global_variables_initializer()
sess.run(init)

# Обучаем
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
train_step = optim.minimize(loss)
for i in range(100):

    summary = sess.run(merged_summary)
    summary_writer.add_summary(summary, i)

    sess.run(train_step)

print(sess.run(output_value))

summary_writer.close()
