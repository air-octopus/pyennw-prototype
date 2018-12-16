# coding=utf-8

from engine import Engine

import neural_network_impl as nn

import collections
import json

class Hasher:
    """
    класс для вычисления хеша графа нейросети
    идея:
        * вводим порядок для нейронов и синапсов. Порядок должен
          (по возможности однозначно) определяться геометрией сети и ее параметрами,
          но не зависеть от возможных случайных перестановок нейронов и синапсов
        * в соответствии с этим порядком сохраняем ключевую информацию
          о всей нейросети в виде строки
        * из строки генерируем md5

    нейроны группируются по значению effective_deepness и в каждой группе сортируются независимо
    (соответственно имеют индекс в группе -- deepness_group_index)

    нейронам приписывается составной идентификатор (effective_deepness, deepness_group_index)

    критерии сортировки нейронов (в порядке значимости):
        * effective_deepness
        * sid (для входных и выходных нейронов)
        * количество входов
        * количество выходов
        * кортеж из отсортированных составных идентификаторов нейронов, связанных с данным
          и находящихся дальше по иерархии

    критерии сортировки синапсов (в порядке значимости):
        * нейрон-источник
        * нейрон-приемник
    """

    # классы для специфического представления нейронов и синапсов
    class NeuronPresentation:   pass
    class SynapsePresentation:  pass

    def __init__(self):
        pass

    def caclulate_hash(self, d : nn.Data):
        neurons  = [Hasher.NeuronPresentation ()] * len(d.neurons)
        synapses = [Hasher.SynapsePresentation()] * len(d.synapses)

        for n, origin in zip(neurons, d.neurons):
            n.origin = origin
            n.sid = ""

        for i in d.input_neurons:
            neurons[i].sid = Engine

        neuron_ind_to_ancestors_inds = collections.defaultdict(list)
        neuron_ind_to_successors_inds = collections.defaultdict(list)
        for s in d.synapses:
            neuron_ind_to_ancestors_inds[s.own].append(s.src)
            neuron_ind_to_successors_inds[s.src].append(s.own)





        pass





















