# coding=utf-8

"""
Методы описания нейронной сети и работы с ней в процессе обучения и/или использования
"""


import neural_network_impl as nn
import collections


class NeuralNetwork:
    """
    Класс, описывающий нейронную сеть.
    Структура нейросети:
        * _data
          массив массивов имеющий смысл массива нейронов и входных данных.
          Каждый нейрон описывается массивом значений, представляющих собой данные аксона.
          Нулевой элемент этого массива -- аккумулятор для текущих вычислений на нейроне.
          Последний элемент -- текущее значение нейрона.
          Самый простейший аксон состоит не менее чем из двух элементов
          Входные данные представляют собой массив из одного элемента
        * _synapses
          массив триплетов, описывающих синапсы. Элементы триплетов имеют следующий смысл:
            * вес синаптической связи
            * индекс нейрона, являющегося входным для данного синапса
            * индекс нейрона, являющегося выходным для синапса
    Note: _synapses определяет конфигурацию нейросети и состояние ее обученности.
          _data определяет оперативные данные.
          Зануление всех элементов _data сбрасывает нейросеть в начальное состояние.
          Зануление _synapses уничтожит нейросеть
    Нейросеть работает тактами. На каждом такте выполняются следующие действия:
        * заполняются входные данные
        * данные аксонов смещаются на один шаг (по направлению к хвосту)
        * начала аксонов, представляющие собой аккумуляторы для вычислений на нейроне зануляются
        * для всех синапсов выполняются вычисления. Результат вычислений аккумулируется в начале соответствующего аксона
        * для всех нейронов выполняется вычисление значения передаточной функции. Источником и результатом является аккумулятор нейрона
        * заполняются выходные данные
    """

    @property
    def data(self):
        return self._data

    def __init__(self, id=0):

        if id == 0:
            self._data = nn.Builder().build_protozoan()
        else:
            self._data = nn.SaveLoad().load(id)

    def save(self):
        # todo: реализовать вычисление времени отклика, качества и приспособленности НС
        return nn.SaveLoad().save(self._data)

    def load_inputs(self, inputs):
        """
        Загрузка входных данных в нейросеть.
        """
        for neuron_ind, val in zip(self.data.input_neurons_inds, inputs):
            self.data.neurons[neuron_ind].axon[0] = val

    def get_outputs(self):
        """
        Выгрузка выходных данных
        """
        return list(self.data.neurons[output_ind].axon[-1] for output_ind in self.data.output_neurons_inds)

    def do_iteration(self, ):
        # todo: реализовать через tensorflow
        pass

    def reset(self):
        """
        Сбросить состояние нейросети в исходное.
        """
        self.data.reset()

    def print_gv(self):
        d = self.data
        neuron_synapses = collections.defaultdict(list)
        for i, s in enumerate(d.synapses):
            neuron_synapses[s.own].append(i)

        neurons_str = []
        for ni, n in enumerate(d.neurons):
            neuron_synapses_str = "{" + "|".join(["<s%d> %.5f" % (si, d.synapses[si].weight) for si in neuron_synapses[ni]]) + "}|" if ni in neuron_synapses else ""
            neurons_str.append('    n%d [label="{%s<r>}"];' % (ni, neuron_synapses_str))


        synapses_str = []
        for si, s in enumerate(d.synapses):
            synapses_str.append('    n%d:r -> n%d:s%d' % (s.src, s.own, si))

        result = '''digraph structs {
    node [shape=record];
    rankdir=LR;
        
%s

%s
}
''' % ('\n'.join(neurons_str), '\n'.join(synapses_str))

        return result
