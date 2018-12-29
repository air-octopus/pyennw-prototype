# coding=utf-8

"""
Вычисления на нейросети.
N!B! в настоящее время поддерживаются только аксоны с длиной 2!!!

Для вычислений используется TensorFlow.

компоненты:
    * in : вектор входных значений, который представлен placeholder'ом
    * a1 : вектор первых элементов аксонов
    * a2 : вектор вторых элементов аксонов
    * w  : вектор весов
    * b  : вектор смещений
    * out: выходные данные
    * p# : разного рода промежуточные данные

Calculator работает в двух режимах:
    * режим обучения НС (см. calculator_trainer.py)
    * потоковый режим вычислений на НС (см. calculator_flow.py)
"""

import tensorflow as tf
import neural_network_impl as nn

class CalculatorBase:

    def __init__(self, d : nn.Data):

        # Исходная нейросеть
        self._data = d

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

        # тензоры весов и смещенй.
        # В режиме тренировки представляют собой экземпляры tf.Variable()
        # В режиме вычислений -- tf.constant
        self._w   = None
        self._b   = None

        worker = [(i, n) for i, n in enumerate(d.neurons) if not n.is_receptor]

        # отображения индексов нейронов в индексы рецепторов и рабочих нейронов
        neuron_ind_to_receptor_ind = {n: r for r,  n in enumerate(d.input_neurons_inds)}
        neuron_ind_to_worker_ind   = {n: w for w, (n, o)  in enumerate(worker)}

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

        self._indices_gather_indicators  = [neuron_ind_to_worker_ind[neuron_ind] for neuron_ind in d.output_neurons_inds]

        # длины векторов в векторизованном представлении нейросети (для одной итерации)
        self._in_len  = len(d.input_neurons_inds )
        self._a_len   = len(worker               )
        self._w_len   = len(d.synapses           )
        self._out_len = len(d.output_neurons_inds)

        #self._a_zeros_init = tf.constant([0] * self._a_len, dtype=tf.float32)

    def _build_iteration_body(self, a2):
        """
        Основное содержимое одной итерации вычислений
        :param a2:  текущее содержимое аксонов рабочих нейронов.
                    Представляется в виде tf-тензора.
                    Используется как вход для вычислений
        :return: кортеж (receptors, a1, indicators)
                    receptors -- плейсхолдеры для входных данных
                    a1 -- результат вычислений на рабочих нейронах (tf-тензоры)
                    indicators -- выходные данные (tf-тензоры)
        """

        # входные данные
        receptors = tf.placeholder(dtype=tf.float32, shape=[self._in_len])

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

        p6 = tf.segment_sum(p5, self._indices_segment_sum_synapses)

        a1 = p6 + self._b

        # todo: добавить функцию активации

        indicators = tf.gather(a1, self._indices_gather_indicators)

        return (receptors, a1, indicators)
