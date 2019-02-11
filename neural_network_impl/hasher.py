# coding=utf-8

from engine import Engine

import neural_network_impl as nn

import collections
import hashlib
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

    нейроны группируются по значению deepness и в каждой группе сортируются независимо
    (соответственно имеют индекс в группе -- deepness_group_index)

    нейронам приписывается составной идентификатор - кортеж cid=(deepness, deepness_group_index)

    критерии сортировки нейронов (в порядке значимости):
        * deepness
        * sid (для входных и выходных нейронов)
        * количество входов
        * количество выходов
        * кортеж из отсортированных составных идентификаторов нейронов, связанных с данным
          и находящихся дальше по иерархии

    критерии сортировки синапсов (в порядке значимости):
        * нейрон-источник
        * нейрон-приемник

    сохраняемая информация
        для нейронов:
            * тип передаточной функции
            * параметры передаточной функции
            * длина аксона
            * глубина
        для синапсов:
            * cid нейрона-источника
            * cid нейрона-приемника
            * вес
    """

    num_format = "%.3e"

    # классы для специфического представления нейронов и синапсов
    class NeuronPresentation:   pass
    class SynapsePresentation:  pass

    def __init__(self):
        pass

    @classmethod
    def caclulate_hash(cls, d : nn.Data):
        neurons  = [cls.NeuronPresentation () for unused in d.neurons ]
        synapses = [cls.SynapsePresentation() for unused in d.synapses]

        for i, (n, origin) in enumerate(zip(neurons, d.neurons)):
            n.origin = origin
            n.origin_ind = i
            n.sid = ""

        input_sids = d.extra_data["input_sids"]
        for input_ind, neuron_ind in enumerate(d.input_neurons_inds):
            neurons[neuron_ind].sid = input_sids[input_ind]

        output_sids = d.extra_data["output_sids"]
        for output_ind, neuron_ind in enumerate(d.output_neurons_inds):
            neurons[neuron_ind].sid = output_sids[output_ind]

        neuron_ind_to_ancestors_inds = collections.defaultdict(list)
        neuron_ind_to_successors_inds = collections.defaultdict(list)
        for s in d.synapses:
            neuron_ind_to_ancestors_inds[s.own].append(s.src)
            neuron_ind_to_successors_inds[s.src].append(s.own)

        deepness_to_neuron_inds = collections.defaultdict(list)
        for n in neurons:
            deepness_to_neuron_inds[n.origin.deepness].append(n.origin_ind)

        deepness_groups = list(deepness_to_neuron_inds.keys())
        deepness_groups.sort()
        deepness_groups.reverse()

        for deepness in deepness_groups:
            for neuron_ind in deepness_to_neuron_inds[deepness]:
                successors_cids = sorted([neurons[i].cid for i in neuron_ind_to_successors_inds[neuron_ind] if d.neurons[i].deepness > deepness])
                neurons[neuron_ind].sort_criterium = (
                    deepness,
                    neurons[neuron_ind].sid,
                    len(neuron_ind_to_ancestors_inds[neuron_ind]),
                    len(neuron_ind_to_successors_inds[neuron_ind]),
                    tuple(successors_cids)
                )
            deepness_to_neuron_inds[deepness].sort(key=lambda x: neurons[x].sort_criterium)
            for deepness_group_index, neuron_ind in enumerate(deepness_to_neuron_inds[deepness]):
                neurons[neuron_ind].cid = (deepness, deepness_group_index)

        neuron_ind_to_cid = {n.origin_ind: n.cid for n in neurons}

        neurons_info = [
            (
                n.origin.transfer_function_type,
                # tuple(Hasher.num_format % p for p in n.origin.transfer_function_params),
                len(n.origin.axon),
                n.origin.deepness
            )
            for n in sorted(neurons, key=lambda o: o.cid)
        ]

        synapses_info = [
            (
                neuron_ind_to_cid[s.src],
                neuron_ind_to_cid[s.own],
                # Hasher.num_format % s.weight
            )
            for s in sorted(d.synapses, key=lambda o: (neuron_ind_to_cid[o.src], neuron_ind_to_cid[o.own]))
        ]

        all_info = {
            "n": neurons_info,
            "s": synapses_info
        }

        return hashlib.md5(json.dumps(all_info, sort_keys=True).encode()).hexdigest()





















