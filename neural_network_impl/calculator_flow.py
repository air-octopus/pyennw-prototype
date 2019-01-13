# coding=utf-8

"""
Вычисления на нейросети.
Потоковый режим вычислений

в потоковом режиме вычислений:
    * есть возможность выполнять произвольное количество итераций
    * нейросеть не разворачивается
    * все вычисления производятся по одной итерации за раз
    * для внутренних синаптических связей (между внутренними нейронами, для которых используется a2)
      при задании нейросети используется placeholder
    * все вычисления выполняются по одной итерации за раз
    * в конце итерации извлекаются данные a2, out. Значения out накапливаются в списке,
      а a2 используется для выполнения следующей итерации
"""

import tensorflow as tf
import neural_network_impl as nn

class CalcFlow(nn.CalculatorBase):
    def __init__(self, d : nn.Data):
        super().__init__(d)

        # текущие значения аксонов (массив вещественных чисел)
        self._a_data = None

        # сохраненные данные
        self._stored_a   = []
        self._stored_out = []

        self._stored_max_count = None

        self._a_data = [0] * self._a_len

        self._w = tf.constant([synapse.weight for synapse in self._data.synapses], dtype=tf.float32)
        # self._b = tf.constant([self._data.neurons[i].bias for i in self._indices_stitch_workers], dtype=tf.float32)
        self._b = tf.constant([neuron.bias for neuron in self._data.neurons if not neuron.is_receptor], dtype=tf.float32)
        self._a2 = tf.placeholder(dtype=tf.float32, shape=[self._a_len])

        self._in, self._a1, self._out = self._build_iteration_body(self._a2)

        self._sess = tf.Session()

    def step(self, receptors):

        feed_data = {
            self._in: receptors,
            self._a2: self._a_data
        }

        result = self._sess.run([self._a1, self._out], feed_dict=feed_data)
        self._a_data = result[0]

        return result[1]

    def multi(self, receptors_array):
        return [self.step(receptors) for receptors in receptors_array]
