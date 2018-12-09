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

class Calculator:

    def __init__(self, neural_network : neural_network.NeuralNetwork):
        self._neural_network = neural_network

        class NeuronExtended:
            def __init__(self, n : nn.Neuron):
                pass

        d = neural_network.data

        # Разделим все нейроны на рецепторы и рабочие нейроны.
        # Компонент 'in' соответствует рецепторам, а компоненты a1 и a2 -- рабочим нейронам
#        self._receptors = [(i, n) for i, n in enumerate(d.neurons) if n.is_receptor]
        self._worker    = [(i, n) for i, n in enumerate(d.neurons) if not n.is_receptor]

        # отображения индексов нейронов в индексы рецепторов и рабочих нейронов
        neuron_ind_to_receptor_ind = {n: r for r,  n      in enumerate(d.input_neurons)}
        neuron_ind_to_worker_ind   = {n: w for w, (n, o)  in enumerate(self._worker)}

        # отсортируем рецепторы в соответствии



        # надо:
        #   * input_ind  -> synapse_ind
        #   * worker_ind -> synapse_ind
        # есть:
        #   * input_ind -> input_sid    (т.к. есть массив input_sid)
        #   * neuron_ind <-> input_sid  (т.к. есть связанные массивы input_neurons и extra_data.input_sid)
        #   * sinapse_ind -> neuron_ind
        #
        # делаем:
        #   * input_ind -> neuron_ind
        #
        # input_sid -> input_ind

        # neuron_ind_to_synapse_ind = defaultdict(list)
        # for synapse_ind, synapse in enumerate(d.synapses):
        #     neuron_ind_to_synapse_ind[synapse.src].append(synapse_ind)
        #
        # input_sid_to_neuron_ind = {}
        # for i, sid in enumerate(d.extra_data["input_sids"]):
        #     input_sid_to_neuron_ind[sid] = d.input_neurons[i]

        synapse_neuron_in_neuron_out_is_receptor = [[synapse_ind, synapse.src, d.neurons[synapse.src].is_receptor] for synapse_ind, synapse in enumerate(d.synapses)]
#        neuron_synapse_indices.sort(key = lambda x: not d.neurons[x[1]].is_receptor)

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

        # # индексы для векторизованного представления нейросети
        #
        # def build_indx_a2_for_in(self):
        #     indx = list({synapse.src for synapse in d.synapses})
        #     indx.sort()
        #     return indx
        #
        # def build_indx_extin_to_w(self):
        #     pass
        #
        # self.indx_a2_for_in = build_indx_a2_for_in()
        # # * indx_extin_to_w : индексы для адаптации расширенного входа к w
        # # * indx_w_to_a1    : индексы для аккумулирования результатов вычислений на синапсах во входах аксонов
        # # * indx_a2_to_out  : индексы для сбора выходных данных

        self._a_zeros_init = tf.constant([0] * self._a1_len)

        self.clear()

    def clear(self):
        nn = self._neural_network

        # все компоненты представлены списками, т.к. для обучени НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.

        self._in  = []
        self._a   = []
        self._w   = []
        self._out = []

        # self._in_shape  = [len(d.input_neurons )]
        # self._a1_shape  = [len(d.neurons       )]
        # self._a2_shape  = [len(d.neurons       )]
        # self._w_shape   = [len(d.synapses      )]
        # self._out_shape = [len(d.output_neurons)]

        self._training_data = Engine.training_data()

    def _add_iteration_for_training(self):
        """
        В режиме тренировки нейросеть разворачивается сразу на несколько итераций.
        :return:
        """
        # номер итерации (в батче)
        it_num = len(self._in)

        # входные данные
        receptors = tf.placeholder(dtype=tf.float32, shape=[self._in_len], name="input_value[%d]" % it_num)

        # текущие выходные значения рабочих нейронов
        a2 = self._a[-1] if len(self._a) > 0 else self._a_zeros_init

        # подготавливаем данные для свертки с массивом весов
        p1 = tf.gather(receptors, self._indices_gather_receptors)
        p2 = tf.gather(a2, self._indices_gather_workers)
        p3 = tf.dynamic_stitch([self._indices_stitch_receptors, self._indices_stitch_workers], [p1, p2])

        a1 = tf.Variable([0] * self._a1_len)
        a1 = tf.scatter_add(a1, self._indices_scater_add_workers, p3)

        indicators = tf.gather(a1, self._indices_gather_indicators)



        pass



















