# coding=utf-8

"""
Вычисления на нейросети.
N!B! в настоящее время поддерживаются только аксоны с длиной 2!!!

Для вычислений используется TensorFlow.

компоненты:
    * in : вектор входных значений, который представлен placeholder'ом
    * a1 : вектор первых элементов аксонов
    * a2 : вектор вторых элементов аксонов
    * w  : вектор весов (variable)
    * out: выходные данные
    * p# : разного рода промежуточные данные

индексы для векторизованного представления НС:
    * indx_a2_for_in  : индексы нейронов, которые используются как вход для синапсов
    * indx_extin_to_w : индексы для адаптации расширенного входа к w
    * indx_w_to_a1    : индексы для аккумулирования результатов вычислений на синапсах во входах аксонов
    * indx_a2_to_out  : индексы для сбора выходных данных

Calculator работает в двух режимах:
    * режим обучения НС
    * режим вычислений на НС

в режиме обучения:
    * изначально задается необходимое количество итераций в одном батче
    * нейросеть разворачивается сразу на все итерации
    * все итерации в одном батче вычисляются как единое целое
    * результатом является массив ответов нейросети
    * требуется функция потерь, которая учитывает сразу все ответы нейросети
    * веса задаются как tf.Variable
    * начальные данные аксонов инициализируются нулями, и в дальнейшем аккумулируют в себе все предыдущие вычисления
в режиме вычислений:
    * есть возможность выполнять произвольное количество итераций
    * нейросеть не разворачивается
    * все вычисления производятся по одной итерации за раз
    * для внутренних синаптических связей (между внутренними нейронами, для которых используется a2)
      при задании нейросети используется placeholder
    * все вычисления выполняются по одной итерации за раз
    * в конце итерации извлекаются данные a2, out. Значения out накапливаются в списке,
      а a2 используется для выполнения следующей итерации

Последовательность:
     ----- init
     * a2  <- zeros
     ----- cycle
  ,->* p1  <- tf.gather(a2)                 # выделяем только те a2, которые используются как входные данные
  |  * p1  <- tf.dynamic_stitch([in, a2])   # расширенные входные данные (вход НС + internal)
  |  * p2  <- tf.gather(p1)                 # адаптируем к w
  |  * p3  <- tf.multiply(p2, w)
     * a1  <- zeros
  |  * a1  <- tf.scatter_add(a1, p3)
  |  * a2  <- a1
  |  * out <- tf.gather(a2)
  `--* loop
"""

import tensorflow as tf
import neural_network
import neural_network_impl as nn
from engine import Engine

from collections import defaultdict

class _Calculator_impl:

    def __init__(self, neural_network : neural_network.NeuralNetwork):
        # Исходная нейросеть
        self._neural_network = neural_network

        # Массивы тензоров векторизованного представления нейросети.
        #
        # В режиме тренировки нейросети входные данные бьются на батчи по несколько записей.
        # Каждой записи соответствует одна итерация работы нейросети и, соответственно,
        # один элемент в этих массивах
        #
        # В режиме простых вычислений все массивы содержат только по одному элементу

        # массив tf-плейсхолдеров для входных данных.
        self._in  = None
        # массив данных в аксонах
        self._a   = None
        # массив выходных данных
        self._out = None

        # тензор весов.
        # В режиме тренировки представляет собой экземпляр tf.Variable()
        # В режиме вычислений -- tf.constant
        self._w   = None

        d = neural_network.data

        # Разделим все нейроны на рецепторы и рабочие нейроны.
        # Компонент 'in' соответствует рецепторам, а компоненты a1 и a2 -- рабочим нейронам
        self._worker    = [(i, n) for i, n in enumerate(d.neurons) if not n.is_receptor]

        # отображения индексов нейронов в индексы рецепторов и рабочих нейронов
        neuron_ind_to_receptor_ind = {n: r for r,  n      in enumerate(d.input_neurons)}
        neuron_ind_to_worker_ind   = {n: w for w, (n, o)  in enumerate(self._worker)}

        synapse_neuron_in_neuron_out_is_receptor = [[synapse_ind, synapse.src, synapse.own, d.neurons[synapse.src].is_receptor] for synapse_ind, synapse in enumerate(d.synapses)]

        info_gather_receptors = [(synapse_ind, neuron_ind_to_receptor_ind[neuron_in_ind]) for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor if is_receptor is True]
        info_gather_workers   = [(synapse_ind, neuron_ind_to_worker_ind  [neuron_in_ind]) for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor if is_receptor is False]

        self._indices_gather_receptors   = [o[1] for o in info_gather_receptors]
        self._indices_gather_workers     = [o[1] for o in info_gather_workers  ]

        self._indices_stitch_receptors   = [o[0] for o in info_gather_receptors]
        self._indices_stitch_workers     = [o[0] for o in info_gather_workers  ]

        info_scatter_add_workers = [(synapse_ind, neuron_ind_to_worker_ind[neuron_out_ind]) for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor]
        info_scatter_add_workers.sort(key=lambda x: x[1])
        self._indices_gather_synapses_for_sum = [o[0] for o in info_scatter_add_workers]
        self._indices_segment_sum_synapses    = [o[1] for o in info_scatter_add_workers]

        self._indices_gather_indicators  = [neuron_ind_to_worker_ind[neuron_ind] for neuron_ind in d.output_neurons]

        # длины векторов в векторизованном представлении нейросети (для одной итерации)
        self._in_len  = len(d.input_neurons )
        self._a1_len  = len(self._worker    )
        self._a2_len  = len(self._worker    )
        self._w_len   = len(d.synapses      )
        self._out_len = len(d.output_neurons)

        self._a_zeros_init = tf.constant([0] * self._a1_len, dtype=tf.float32)

    def clear(self):
        # все компоненты представлены списками, т.к. для обучения НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.
        self._in  = []
        self._a   = []
        self._out = []
        self._w   = None

    def _build_iteration_body(self, a2):
        """
        Основное содержимое одной итерации вычислений
        :param a2:  текущее содержимое аксонов рабочих нейронов.
                    Представляется в виде tf-тензора.
                    Используется как вход для вычислений
        :return: кортеж (receptors, a1, indicators)
                    receptors -- плейсхолдеры для входных данных
                    a1 -- результат вычислений на рабочих нейронах
                    indixcators -- выходные данные (tf-тензоры)
        """
        # номер итерации (в батче)
        it_num = len(self._in)

        # входные данные
        receptors = tf.placeholder(dtype=tf.float32, shape=[self._in_len], name="input_value_%d" % it_num)

        # подготавливаем данные для свертки с массивом весов
        stitch_ind = []
        stitch_src = []

        if len(self._indices_gather_receptors) > 0 and len(self._indices_stitch_receptors) > 0:
            p1 = tf.gather(receptors, self._indices_gather_receptors)
            stitch_ind.append(self._indices_stitch_receptors)
            stitch_src.append(p1)
        else:
            assert len(self._indices_gather_receptors) + len(self._indices_stitch_receptors) == 0, "Indices can be empty only together"

        if len(self._indices_gather_workers) > 0 and len(self._indices_stitch_workers) > 0:
            p2 = tf.gather(a2, self._indices_gather_workers)
            stitch_ind.append(self._indices_stitch_workers)
            stitch_src.append(p2)
        else:
            assert len(self._indices_gather_workers) + len(self._indices_stitch_workers) == 0, "Indices can be empty only together"

        p3 = tf.dynamic_stitch(stitch_ind, stitch_src)

        p4 = p3 * self._w

        p5 = tf.gather(p4, self._indices_gather_synapses_for_sum)

        a1 = tf.segment_sum(p5, self._indices_segment_sum_synapses)

        # todo: добавить функцию активации

        indicators = tf.gather(a1, self._indices_gather_indicators)

        return (receptors, a1, indicators)

class Trainer(_Calculator_impl):
    def __init__(self, neural_network : neural_network.NeuralNetwork):
        super().__init__(neural_network)

        # Массивы тензоров векторизованного представления нейросети.
        #
        # В режиме тренировки нейросети входные данные бьются на батчи по несколько записей.
        # Каждой записи соответствует одна итерация работы нейросети и, соответственно,
        # один элемент в этих массивах

        # массив tf-плейсхолдеров для входных данных.
        self._in  = None
        # массив данных в аксонах
        self._a   = None
        # массив выходных данных
        self._out = None

        # тензор весов.
        # В режиме тренировки представляет собой экземпляр tf.Variable()
        self._w   = None

        # функция потерь, которая будет минимизироваться в процессе тренировки
        self._loss = None

    def init(self, iterations_count):
        assert iterations_count > 0

        # все компоненты представлены списками, т.к. для обучения НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.
        self._in      = []
        self._a       = []
        self._out     = []
        self._desired = []
        self._w = tf.Variable([synapse.weight for synapse in self._neural_network.data.synapses], dtype=tf.float32)

        for i in range(iterations_count):
            self._add_iteration()

        self._loss = None
        skip = (int)(self._neural_network._data._response_time)
        for out in self._out:
            desired = tf.placeholder(dtype=tf.float32, shape=out.shape)
            self._desired.append(desired)
            if skip > 0:
                skip -= 1
                continue
            dloss = tf.reduce_mean(tf.squared_difference(out, desired))
            self._loss = dloss if self._loss is None else self._loss + dloss

    def training(self, steps_count):
        sess = tf.Session()

        # todo: удалить все ненужное

        # todo: learning_rate должен быть или в настройках или вычисляться адаптивно
        #optim = tf.train.GradientDescentOptimizer(learning_rate=0.0025)  # Оптимизатор
        optim = tf.train.AdamOptimizer(learning_rate=0.25)  # Оптимизатор

        grads_and_vars = optim.compute_gradients(self._loss)

        train_step = optim.minimize(self._loss)

        loss_val = -1
        sess.run(tf.global_variables_initializer())
        for batch in self._training_set_batch(steps_count):
            feed_data = {}
            for tf_in, tf_desired, training_set in zip(self._in, self._desired, batch):
                feed_data[tf_in] = training_set.data_in
                feed_data[tf_desired] = training_set.data_out
            result = sess.run([train_step, grads_and_vars, self._loss, self._out], feed_dict=feed_data)
            loss_val = result[2]
            # all_data = [
            #     self._loss,
            #     self._w,
            #     self._in,
            #     self._a,
            #     self._out,
            #     self._desired
            # ]
            #
            # all_data = sess.run(all_data, feed_dict=feed_data)
            pass

        weights = sess.run(self._w)
        for synapse, w in zip(self._neural_network.data.synapses, weights):
            synapse.weight = w

        # todo: вынести отдельно весь код оценки нейросети
        # сейчас для простоты используем вычисленное значение функции потерь,
        # однако в дальнейшем надо будет оценивать и время реакции и разрешающую способность и проч
        self._neural_network.data._quality = loss_val

    def _add_iteration(self):
        """
        В режиме тренировки нейросеть разворачивается сразу на несколько итераций.
        """
        # текущие выходные значения рабочих нейронов
        a2 = self._a[-1] if len(self._a) > 0 else self._a_zeros_init

        receptors, a1, indicators = self._build_iteration_body(a2)

        self._in .append(receptors )
        self._a  .append(a1        )
        self._out.append(indicators)

    def _training_set_batch(self, steps_count):
        batch = None
        batch_len = len(self._in)
        step = 0
        for ts in Engine.training_data().training_set_loopped():
            if batch is None:
                batch_len = len(self._in)
                batch = []
                #batch = [[], []]
            batch.append(ts)
            # batch[0].append(ts.data_in)
            # batch[1].append(ts.data_out)
            if len(batch) >= batch_len:
                yield batch
                step += 1
                if step >= steps_count:
                    return
                batch = None
