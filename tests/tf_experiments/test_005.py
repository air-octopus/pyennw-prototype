##################################################################
## http://wiki.programstore.ru/python-tensorflow-dlya-chajnikov-chast-1/
#
# Пример использования обучения tensorflow-графа нейросети на простом наборе тестовых данных.
# Граф: два нейрона первого слоя и один нейрон второго
#       у нейронов первого слоя есть смещение
#       передаточная функция -- relu
#
# Требуемое значение на выходе: (a xor b)
# Количество итераций: 200
#
# Для визуализации предполагаем использовать tensorboard

import datetime
import tensorflow as tf


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
        # tf.summary.histogram('histogram', var)


def tensor_to_list(t):
    if t is None:
        return []
    l = []
    if t.shape.ndims > 0:
        for subt in (t[i] for i in range(t.shape.dims[0])):
            l.extend(tensor_to_list(subt))
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
# смещения нейронов (только для первой сети -- 2 нейрона)
bias_lay_1 = tf.Variable(bias_initializer([2]), name="bias_lay_1")
# # bias_lay_2 = tf.Variable(bias_initializer([1]), name="bias_lay_2")

# weight_lay_1 = tf.Variable([[1, 1], [1, 1]], name="weight_lay_1", dtype=tf.float32)
# # weight_lay_2 = tf.Variable([[1], [-2]], name="weight_lay_2", dtype=tf.float32)
# weight_lay_2 = tf.Variable([1, -2], name="weight_lay_2", dtype=tf.float32)
# weight_lay_2 = tf.reshape(weight_lay_2, shape=(2, 1))
# bias_lay_1 = tf.Variable([0, -1], name="bias_lay_1", dtype=tf.float32)
# # bias_lay_2 = tf.Variable([0], name="bias_lay_2", dtype=tf.float32)
# # bias_lay_2 = tf.Variable(0, name="bias_lay_2", dtype=tf.float32)

# # weight_lay_1 = tf.Variable([[1.178932, 1.072376], [0.926467, 1.3678287]], name="weight_lay_1", dtype=tf.float32)
# weight_lay_2 = tf.Variable([0.8737834, -2.3245216], name="weight_lay_2", dtype=tf.float32)
# # bias_lay_1 = tf.Variable([0, -1], name="bias_lay_1", dtype=tf.float32)
#
# weight_lay_1 = tf.constant([[1, 1], [1, 1]], name="weight_lay_1", dtype=tf.float32)
# # weight_lay_2 = tf.constant([1, -2], name="weight_lay_2", dtype=tf.float32)
# bias_lay_1 = tf.constant([0, -1], name="bias_lay_1", dtype=tf.float32)
#
# weight_lay_2 = tf.reshape(weight_lay_2, shape=(2, 1))

variable_summaries('weight_lay_1', weight_lay_1)
variable_summaries('weight_lay_2', weight_lay_2)
variable_summaries('bias_lay_1', bias_lay_1)
# variable_summaries('bias_lay_2', bias_lay_2)

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
a1 = tf.matmul(input_value, weight_lay_1)
a2 = a1 + bias_lay_1
a3 = tf.nn.relu(a2)
a = a3 #tf.reshape(a3, shape=(2))
b1 = tf.matmul(a, weight_lay_2)
b1 = tf.reshape(b1, shape=[tf.shape(b1)[0]])
b2 = b1 #+ bias_lay_2
b3 = tf.nn.relu(b2)
b = b3
# a = tf.nn.relu(tf.matmul(input_value, weight_lay_1) + bias_lay_1)
# b = tf.nn.relu(tf.matmul(a          , weight_lay_2) + bias_lay_2)
# a = tf.sigmoid(tf.matmul(input_value, weight_lay_1) + bias_lay_1)
# b = tf.sigmoid(tf.matmul(a          , weight_lay_2) + bias_lay_2)

# tf.summary.histogram('a', a)
# tf.summary.histogram('b', b)

# a = tf.sigmoid(weight_lay_1[0] * input_value + bias_lay_1)
# b = tf.sigmoid(weight_lay_1[1] * input_value + bias_lay_1)
# c = tf.sigmoid(weight_lay_2[2] * list([a, b])+ bias_lay_2)

output_value = b

# создадим сессию
sess = tf.Session()

# Создаем оптимизиатор
#desired_output_value = inout_value[2]
# loss1 = tf.squared_difference(output_value, desired_value)
# loss2 = tf.reduce_mean(loss1)
# loss = loss2
loss = tf.reduce_mean(tf.squared_difference(output_value, desired_value))
variable_summaries('loss', loss)
# loss = (output_value - desired_output_value) ** 2  # Функция ошибки
optim = tf.train.GradientDescentOptimizer(learning_rate=0.25)  # Оптимизатор
grads_and_vars = optim.compute_gradients(loss)
for gav in grads_and_vars:
    for o in tensor_to_list(gav[0]):
        tf.summary.scalar('grad-' + gav[1].name + '-' + o.name, o)
    for o in tensor_to_list(gav[1]):
        tf.summary.scalar('var-' + gav[1].name + '-'  + o.name, o)

#    ooo0 = tensor_to_list(gav[0])
#    ooo1 = tensor_to_list(gav[1])
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
# data_out = [[0], [1], [1], [0]]

merged = tf.summary.merge_all()

file_writer = tf.summary.FileWriter('/home/maksko/-/Temp/001/' + str(datetime.datetime.now()), graph)

# Обучаем
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
train_step = optim.minimize(loss)
feed_data={input_value: data_in, desired_value: data_out}
for i in range(200):
    summary = sess.run(merged, feed_dict=feed_data)
    file_writer.add_summary(summary, i)

    # cur_a1    = sess.run(   a1, feed_dict=feed_data)
    # cur_a2    = sess.run(   a2, feed_dict=feed_data)
    # cur_a3    = sess.run(   a3, feed_dict=feed_data)
    # cur_a     = sess.run(   a , feed_dict=feed_data)
    # cur_b1    = sess.run(   b1, feed_dict=feed_data)
    # cur_b2    = sess.run(   b2, feed_dict=feed_data)
    # cur_b3    = sess.run(   b3, feed_dict=feed_data)
    # cur_b     = sess.run(   b , feed_dict=feed_data)
    # cur_loss  = sess.run(loss , feed_dict=feed_data)
    # cur_loss1 = sess.run(loss1, feed_dict=feed_data)
    # cur_loss2 = sess.run(loss2, feed_dict=feed_data)








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
