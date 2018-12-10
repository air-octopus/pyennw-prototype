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

        # Внутренние данные класса Calculator зависятот того

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

        synapse_neuron_in_neuron_out_is_receptor = [[synapse_ind, synapse.src, d.neurons[synapse.src].is_receptor] for synapse_ind, synapse in enumerate(d.synapses)]

        info_gather_receptors = [(synapse_ind, neuron_ind_to_receptor_ind[neuron_in_ind]) for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor if is_receptor is True]
        info_gather_workers   = [(synapse_ind, neuron_ind_to_worker_ind  [neuron_in_ind]) for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor if is_receptor is False]

        self._indices_gather_receptors   = [o[1] for o in info_gather_receptors]
        self._indices_gather_workers     = [o[1] for o in info_gather_workers  ]

        self._indices_stitch_receptors   = [o[0] for o in info_gather_receptors]
        self._indices_stitch_workers     = [o[0] for o in info_gather_workers  ]

        self._indices_scater_add_workers = [neuron_out_ind for synapse_ind, neuron_in_ind, neuron_out_ind, is_receptor in synapse_neuron_in_neuron_out_is_receptor if is_receptor is False]

        self._indices_gather_indicators  = [neuron_ind_to_worker_ind[neuron_ind] for neuron_ind in d.output_neurons]

        # длины векторов в векторизованном представлении нейросети (для одной итерации)
        self._in_len  = len(d.input_neurons )
        self._a1_len  = len(self._worker    )
        self._a2_len  = len(self._worker    )
        self._w_len   = len(d.synapses      )
        self._out_len = len(d.output_neurons)

        self._a_zeros_init = tf.constant([0] * self._a1_len)

    def clear(self):
#        nn = self._neural_network

        # все компоненты представлены списками, т.к. для обучения НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.

        self._in  = []
        self._a   = []
        self._out = []
        self._w   = None
    #
    # def init_for_training(self, iterations_count):
    #     self.clear()
    #     self._w = tf.Variable([synapse.weight for synapse in self._neural_network.data.synapses])
    #
    # def _add_iteration_for_training(self):
    #     # текущие выходные значения рабочих нейронов
    #     a2 = self._a[-1] if len(self._a) > 0 else self._a_zeros_init
    #
    #     self.__add_iteration_body(a2)

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
        receptors = tf.placeholder(dtype=tf.float32, shape=[self._in_len], name="input_value[%d]" % it_num)

        # подготавливаем данные для свертки с массивом весов
        p1 = tf.gather(receptors, self._indices_gather_receptors)
        p2 = tf.gather(a2, self._indices_gather_workers)
        p3 = tf.dynamic_stitch([self._indices_stitch_receptors, self._indices_stitch_workers], [p1, p2])

        a1 = tf.Variable([0] * self._a1_len, trainable=False)
        a1 = tf.scatter_add(a1, self._indices_scater_add_workers, p3)

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

    def init(self, iterations_count):
        # все компоненты представлены списками, т.к. для обучения НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.
        self._in  = []
        self._a   = []
        self._out = []
        self._w = tf.Variable([synapse.weight for synapse in self._neural_network.data.synapses])
        for i in range(iterations_count):
            self._add_iteration()

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

    def training(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        pass
