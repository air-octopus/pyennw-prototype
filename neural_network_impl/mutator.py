# coding=utf-8

from engine import Engine

import neural_network_impl as nn

import itertools
import random

import collections
# import hashlib
# import json

class Mutator:

    @classmethod
    def mutate(cls, data : nn.Data):
        counter = 0
        hash = data.hash
        while data.hash == hash:
            cls._do_mutate(data)
            counter += 1
        pass

    @classmethod
    def _do_mutate(cls, data : nn.Data):
        d = data
        r = random.Random()
        c = Engine.config()

        receptors_list = [ d.neurons[i] for i in d.input_neurons_inds ]
        indicators_list = [ d.neurons[i] for i in d.output_neurons_inds ]

        neurons = set(d.neurons)
        receptors = set(receptors_list)
        indicators = set(indicators_list)
        workers = { n for n in d.neurons if n not in receptors | indicators }
        synapses = { nn.Synapse(d.neurons[s.src], d.neurons[s.own], s.weight) for s in d.synapses }

        connections = { (s.src, s.own) for s in synapses }

        neuron_to_synapses = collections.defaultdict(set)
        for s in synapses:
            neuron_to_synapses[s.src].add(s)
            neuron_to_synapses[s.own].add(s)

        # удаление нейронов и синапсов

        neurons_to_del = { n for n in workers if r.uniform(0, 1) < c.mutator_neuron_deleting_probability_factor }
        synapses_to_del = { s for s in synapses if r.uniform(0, 1) < c.mutator_synapse_deleting_probability_factor }
        for n in neurons_to_del:
            synapses_to_del |= neuron_to_synapses[n]

        neurons -= neurons_to_del
        synapses -= synapses_to_del

        del workers, neurons_to_del, synapses_to_del
        del neuron_to_synapses

        # добавление нейронов и синапсов

        neurons_to_add = set()
        synapses_to_add = set()
        for n1, n3 in itertools.product(list(neurons), list(neurons - receptors)):
            if r.uniform(0, 1) < c.mutator_synapse_adding_probability_factor:
                if (n1, n3) not in connections:
                    synapses_to_add.add(nn.Synapse(n1, n3, 1))
                    connections.add((n1, n3))
            if r.uniform(0, 1) < c.mutator_neuron_adding_probability_factor:
                n2 = nn.Neuron([0, 0], nn.Type.relu, ())
                neurons_to_add.add(n2)
                synapses_to_add.add(nn.Synapse(n1, n2, 1))
                synapses_to_add.add(nn.Synapse(n2, n3, 1))

        neurons |= neurons_to_add
        synapses |= synapses_to_add

        del neurons_to_add, synapses_to_add
        del n1, n3

        neurons_list = list(neurons)
        synapses_list = list(synapses)

        neuron_to_ind = { n: i for i, n in enumerate(neurons_list) }

        for s in synapses_list:
            s.src = neuron_to_ind[s.src]
            s.own = neuron_to_ind[s.own]

        d._input_neurons_inds = [ neuron_to_ind[n] for n in receptors_list ]
        d._output_neurons_inds = [ neuron_to_ind[n] for n in indicators_list ]
        d._neurons = neurons_list
        d._synapses = synapses_list

        nn.Estimator.fill_all(d)
