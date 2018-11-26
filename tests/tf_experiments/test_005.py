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
# Согласно книге "Глубокое обучение" (Ян Гудфеллоу и проч.) решением является следующий набор значений:
#   * веса первого слоя: [[1, 1], [1, 1]]
#   * смещение первого слоя: [0, -1]
#   * веса второго слоя: [1, -2]
#   * смещение второго слоя: 0
#
# Для визуализации предполагаем использовать tensorboard
#
# Некоторые выводы:
#   * при использовании инициализаторов каждый запуск скрипта переменные инициализируются различными случайными величинами
#   * при использовании RELU в качестве передаточной функциинередка ситуация, когда
#     конечные или промежуточные значения становятся отрицательными.
#     В этом случае RELU обращается в 0 и соответствующие градиенты также зануляются.
#     В силу этого механизм градиентного спуска не работает.
#   * кроме решения, приведенного выше существует множество других решений,
#     которые также могут быть найдены при обучении нейросети.

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
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)


def tensor_to_list(t):
    """
    Представляем тензор в виде простого линейного списка его компонентов.
    По тем измерениям для которых их длина неизвестна производим усреднение.
    Применяется для добавления всех данных в summaries
    """
    if t is None:
        return []
    l = []
    if t.shape.ndims > 0:
        if t.shape.dims[0].value is None:
            l.extend(tensor_to_list(tf.reduce_mean(t, axis=0)))
        else:
            for subt in (t[i] for i in range(t.shape.dims[0])):
                l.extend(tensor_to_list(subt))
    else:
        return [t]
    return l


def tensor_summaries(scope, var):
    with tf.name_scope(scope):
        for o in tensor_to_list(var):
            tf.summary.scalar(o.name, o)


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

# "Эталонное" решение, согласно книге "Глубокое обучение" (Ян Гудфеллоу и проч.)
# weight_lay_1 = tf.Variable([[1, 1], [1, 1]], name="weight_lay_1", dtype=tf.float32)
# # weight_lay_2 = tf.Variable([[1], [-2]], name="weight_lay_2", dtype=tf.float32)
# weight_lay_2 = tf.Variable([1, -2], name="weight_lay_2", dtype=tf.float32)
# weight_lay_2 = tf.reshape(weight_lay_2, shape=(2, 1))
# bias_lay_1 = tf.Variable([0, -1], name="bias_lay_1", dtype=tf.float32)

# "Эталонное" решение с некоторым смещением (для иллюстрации процесса обучения)
# # weight_lay_1 = tf.Variable([[1.178932, 1.072376], [0.926467, 1.3678287]], name="weight_lay_1", dtype=tf.float32)
# weight_lay_2 = tf.Variable([0.8737834, -2.3245216], name="weight_lay_2", dtype=tf.float32)
# weight_lay_2 = tf.reshape(weight_lay_2, shape=(2, 1))
# # bias_lay_1 = tf.Variable([0, -1], name="bias_lay_1", dtype=tf.float32)
# weight_lay_1 = tf.constant([[1, 1], [1, 1]], name="weight_lay_1", dtype=tf.float32)
# # weight_lay_2 = tf.constant([1, -2], name="weight_lay_2", dtype=tf.float32)
# bias_lay_1 = tf.constant([0, -1], name="bias_lay_1", dtype=tf.float32)

variable_summaries('weight_lay_1', weight_lay_1)
variable_summaries('weight_lay_2', weight_lay_2)
variable_summaries('bias_lay_1', bias_lay_1)

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
# b3 = tf.nn.sigmoid(b2)
b = b3

output_value = b

# создадим сессию
sess = tf.Session()

# Создаем оптимизиатор
loss = tf.reduce_mean(tf.squared_difference(output_value, desired_value))
variable_summaries('loss', loss)
optim = tf.train.GradientDescentOptimizer(learning_rate=0.25)  # Оптимизатор
grads_and_vars = optim.compute_gradients(loss)
for gav in grads_and_vars:
    scope = gav[1].name.replace(":", "_")
    tensor_summaries(scope + '_grad', gav[0])
    tensor_summaries(scope + '_var' , gav[1])

tensor_summaries('a1', a1)
tensor_summaries('a2', a2)
tensor_summaries('a3', a3)
tensor_summaries('b1', b1)
tensor_summaries('b2', b2)
tensor_summaries('b3', b3)

# Инициализируем переменные
init = tf.global_variables_initializer()
sess.run(init)

data_in = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]
data_out = [0, 1, 1, 0]

merged = tf.summary.merge_all()

file_writer = tf.summary.FileWriter('/home/maksko/-/Temp/001/' + str(datetime.datetime.now()), graph)

# Обучаем
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

    sess.run(train_step, feed_dict=feed_data)
