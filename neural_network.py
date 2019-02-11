# coding=utf-8

"""
Методы описания нейронной сети и работы с ней в процессе обучения и/или использования
"""

from engine import Engine
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

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = nn.Trainer(self.data)
        return self._trainer

    @property
    def calculator(self):
        if self._calculator is None:
            self._calculator = nn.CalcFlow(self.data)
        return self._calculator

    @classmethod
    def from_json(cls, json_str):
        new_nn = NeuralNetwork()
        new_nn.data.deserialize_json(json_str)
        return new_nn

    def __init__(self, id=0):

        if id == 0:
            self._data = nn.Builder().build_protozoan()
        else:
            self._data = nn.SaveLoad().load(id)

        self._trainer = None
        self._calculator = None

    def save(self):
        return nn.SaveLoad().save(self._data)

    def load_inputs(self, inputs):
        """
        Загрузка входных данных в нейросеть.
        """
        for neuron_ind, val in zip(self.data.input_neurons_inds, inputs):
            self.data.neurons[neuron_ind].axon[0] = val

    def mutate(self):
        nn.Mutator.mutate(self.data)
        self._trainer = None
        self._calculator = None

    def train(self, steps_count):
        # todo: steps_count необходимо оценивать в процессе тренировки (см. "ранняя остановка" в Гудфеллоу etc, 2018)
#        try:
            trainer = self.trainer
            trainer.init(self.data.effective_deepness * 3)
            trainer.training(steps_count)
#        except:
#            pass

    def estimate(self, steps_count):
        # todo: steps_count необходимо брать из исходных данных
        response_time = 0
        quality = 0
        adaptability = 0
        deepness = self.data.effective_deepness
        for step, batch in zip(range(steps_count), Engine.training_data().training_set_batch_gen(deepness * 17)):
            data_in = [o.data_in for o in batch]
            data_desired = [o.data_out for o in batch][:deepness * 5]
            o = nn.Estimator.estimate_adaptability(self.data, self.calculator, data_in, data_desired)
            response_time += o[0]
            quality       += o[1]
            adaptability  += o[2]
        self.data._response_time = response_time / steps_count
        self.data._quality       = quality       / steps_count
        self.data._adaptability  = adaptability  / steps_count

    def get_outputs(self):
        """
        Выгрузка выходных данных
        """
        return list(self.data.neurons[output_ind].axon[-1] for output_ind in self.data.output_neurons_inds)

    def print_gv(self):
        d = self.data
        neuron_synapses = collections.defaultdict(list)
        for i, s in enumerate(d.synapses):
            neuron_synapses[s.own].append(i)

        input_neurons_to_neuron_ind = {d.neurons[i]: i for i in d.input_neurons_inds}
        output_neurons_to_neuron_ind = {d.neurons[i]: i for i in d.output_neurons_inds}
        input_neurons_to_label = {n: d.extra_data["input_sids"][i] for i, n in enumerate(d.input_neurons)}
        output_neurons_to_label = {n: d.extra_data["output_sids"][i] for i, n in enumerate(d.output_neurons)}
        worker_neurons_to_neuron_ind = {n: i for i, n in enumerate(d.neurons) if n not in input_neurons_to_neuron_ind}

        input_rank_str = []
        output_rank_str = []

        neurons_str = []
        neurons_str.append(
            '    info [label="{{'
            'id|'
            'parent_id|'
            'effective_deepness|'
            'response_time|'
            'resolving_ability|'
            'quality|'
            'adaptability'
            '}|{'
            '%d|%d|%g|%g|%g|%g|%g'
            '}}"];' % (
            d.id,
            d.parent_id,
            d._effective_deepness,
            d._response_time,
            d._resolving_ability,
            d._quality,
            d._adaptability,
        ))

        for n, ni in worker_neurons_to_neuron_ind.items():
            neuron_synapses_str = "{" + "|".join(["<s%d> %.5f" % (si, d.synapses[si].weight) for si in neuron_synapses[ni]]) + "}|" if ni in neuron_synapses else ""
            neurons_str.append('    n%d [label="{%s<r> %.5f}"];' % (ni, neuron_synapses_str, d.neurons[ni].bias))

        synapses_str = []
        for si, s in enumerate(d.synapses):
            synapses_str.append('    n%d:r -> n%d:s%d' % (s.src, s.own, si))

        for n, l in input_neurons_to_label.items():
            ni = input_neurons_to_neuron_ind[n]
            neurons_str.append('    n%d [label="{<r> %s}", style=filled, fillcolor="0.5 0.3 1"];' % (ni, l))
            input_rank_str.append('n%d' % ni)

        for n, l in output_neurons_to_label.items():
            ni = output_neurons_to_neuron_ind[n]
            neurons_str.append('    out%d [label="{%s}", style=filled, fillcolor="0.2 0.3 1"];' % (ni, l))
            synapses_str.append('    n%d:r -> out%d' % (ni, ni))
            output_rank_str.append('out%d' % ni)

        result = '''digraph structs {
    node [shape=record];
    rankdir=LR;
        
%s

    { rank = same; %s}
    { rank = same; %s}

%s
}
''' % ('\n'.join(neurons_str), ', '.join(input_rank_str), ', '.join(output_rank_str), '\n'.join(synapses_str))

        return result
