# coding=utf-8

##################################################################
## http://wiki.programstore.ru/python-tensorflow-dlya-chajnikov-chast-1/
#
# Обучение функции OR.
# Граф: два нейрона первого слоя и один нейрон второго
#       у каждого нейрона есть смещение
#       передаточная функция -- сигмоида
#
# Требуемое значение на выходе: 2
# Количество итераций: 100

import tensorflow as tf
import matplotlib.pyplot as plt


def variable_summaries(scope, var):
    """
    https://www.tensorflow.org/guide/summaries_and_tensorboard
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries_' + scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def expand_tensor(t):
    l = []
    if t.shape.ndims > 0:
        for subt in (t[i] for i in range(t.shape.dims[0])):
            l.extend(expand_tensor(subt))
    else:
        return [t]
    return l


# Создадим граф
graph = tf.get_default_graph()

# инициализаторы
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# входные значения (по 2 элемента на каждый запуск нейросети)
# и эталонное результирующее значение
# итого -- 3 числа
input_value = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="input_value")
desired_value = tf.placeholder(dtype=tf.float32, shape=[None], name="desired_value")

# веса синапсов (2 входа * 2 нейрона + 2 входа на нейрон второй сети)
weight_lay_1 = tf.Variable(weight_initializer([2, 2]), name="weight_lay_1")
weight_lay_2 = tf.Variable(weight_initializer([2, 1]), name="weight_lay_2")

bias_lay_1 = tf.Variable(bias_initializer([2]), name="bias_lay_1")
bias_lay_2 = tf.Variable(bias_initializer([1]), name="bias_lay_2")

variable_summaries('weight_lay_1', weight_lay_1)
variable_summaries('weight_lay_2', weight_lay_2)
variable_summaries('bias_lay_1', bias_lay_1)
variable_summaries('bias_lay_2', bias_lay_2)

# weight_lay_1 = [
#     [tf.Variable(0), tf.Variable(0)],   # нейрон 1 (первый слой)
#     [tf.Variable(0), tf.Variable(0)],   # нейрон 2 (первый слой)
# ]
# weight_lay_2 = [
#     [tf.Variable(0), tf.Variable(0)]    # нейрон 3 (второй слой)
# ]

# смещения для трех нейронов
# bias_lay_1 = [tf.Variable(0), tf.Variable(0)]
# bias_lay_2 = [tf.Variable(0)]

# выходное значение
# a = tf.nn.relu(tf.matmul(input_value, weight_lay_1) )
# b = tf.nn.relu(tf.matmul(a          , weight_lay_2) )
a = tf.nn.relu(tf.matmul(input_value, weight_lay_1) + bias_lay_1)
b = tf.nn.relu(tf.matmul(a          , weight_lay_2) + bias_lay_2)
# a = tf.sigmoid(tf.matmul(input_value, weight_lay_1) + bias_lay_1)
# b = tf.sigmoid(tf.matmul(a          , weight_lay_2) + bias_lay_2)

tf.summary.histogram('a', a)
tf.summary.histogram('b', b)

# a = tf.sigmoid(weight_lay_1[0] * input_value + bias_lay_1)
# b = tf.sigmoid(weight_lay_1[1] * input_value + bias_lay_1)
# c = tf.sigmoid(weight_lay_2[2] * list([a, b])+ bias_lay_2)

output_value = b

# создадим сессию
sess = tf.Session()

# Создаем оптимизиатор
#desired_output_value = inout_value[2]
loss = tf.reduce_mean(tf.squared_difference(output_value, desired_value))
variable_summaries('loss', loss)
# loss = (output_value - desired_output_value) ** 2  # Функция ошибки
optim = tf.train.GradientDescentOptimizer(learning_rate=0.25)  # Оптимизатор
grads_and_vars = optim.compute_gradients(loss)
for gav in grads_and_vars:
    for o in expand_tensor(gav[0]):
        tf.summary.scalar('grad-' + gav[1].name + '-' + o.name, o)
    for o in expand_tensor(gav[1]):
        tf.summary.scalar('var-' + gav[1].name + '-'  + o.name, o)

#    ooo0 = expand_tensor(gav[0])
#    ooo1 = expand_tensor(gav[1])
#
#    tf.summary.tensor_summary('grad-' + gav[1].name, gav[0])
#    tf.summary.tensor_summary('var-' + gav[1].name, gav[1])
#    tf.summary.scalar('grad-' + gav[1].name, gav[0])
#    tf.summary.scalar('var-' + gav[1].name, gav[1])

# Инициализируем переменные
init = tf.global_variables_initializer()
sess.run(init)

#    # задаем начальные массивы
#    x = []
#    y = []
#    w10 = []
#    w11 = []
#    w20 = []
#    w21 = []
#    w30 = []
#    w31 = []
#    b1 = []
#    b2 = []
#    b3 = []

data_in = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]
data_out = [0, 1, 1, 0]

merged = tf.summary.merge_all()

file_writer = tf.summary.FileWriter('/home/maksko/-/temp/001', graph)

# Обучаем
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
train_step = optim.minimize(loss)
feed_data={input_value: data_in, desired_value: data_out}
for i in range(200):
    summary = sess.run(merged, feed_dict=feed_data)
    file_writer.add_summary(summary, i)

    sess.run(train_step, feed_dict=feed_data)
    # x.append(i)
    # y.append(sess.run(loss, feed_dict=feed_data))
    # o = sess.run(weight_lay_1)
    # w10.append(o[0][0])
    # w11.append(o[0][1])
    # w20.append(o[1][0])
    # w21.append(o[1][1])
    # o = sess.run(weight_lay_2)
    # w30.append(o[0][0])
    # w31.append(o[1][0])
    # o = sess.run(bias_lay_1)
    # b1.append(o[0])
    # b2.append(o[1])
    # o = sess.run(bias_lay_2)
    # b3.append(o[0])

#print(sess.run(output_value, feed_dict=feed_data))

#    # Строим график
#    fig = plt.figure()
#
#    plt.plot(x, y  )
#    plt.plot(x, w10)
#    plt.plot(x, w11)
#    plt.plot(x, w20)
#    plt.plot(x, w21)
#    plt.plot(x, w30)
#    plt.plot(x, w31)
#    # plt.plot(x, b1 )
#    # plt.plot(x, b2 )
#    # plt.plot(x, b3 )
#
#    # Отображаем заголовки и подписи осей
#    plt.title('График функции')
#    plt.ylabel('Ось Y')
#    plt.xlabel('Ось X')
#    plt.grid(True)
#    # plt.show()
#
#    # результат:
#    #   1.9928955   -- результирующее значение после обучения
#    #   + график, демонстрирующимй уменьшение ошибки
