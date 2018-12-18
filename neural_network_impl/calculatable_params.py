# coding=utf-8

import neural_network_impl as nn

import collections

class CalculatableParams:

    @classmethod
    def fill_all(cls, d : nn.Data):
        cls.fill_deepness(d)
        cls.fill_hash(d)

    @classmethod
    def fill_deepness(cls, d : nn.Data):

        neuron_to_successors = collections.defaultdict(list)
        for s in d.synapses:
            neuron_to_successors[d.neurons[s.src]].append(d.neurons[s.own])

        input_neurons = d.input_neurons
        for n in input_neurons:
            n.deepness = 0

        neurons_processed = set(input_neurons)
        processing_queue = input_neurons
        max_deepness = 0
        while len(processing_queue) > 0:
            n = processing_queue.pop(0)
            deepness = n.deepness
            for succ in neuron_to_successors[n]:
                if succ not in neurons_processed:
                    succ.deepness = deepness + 1
                    if max_deepness <= deepness: max_deepness = deepness + 1
                    processing_queue.append(succ)
                    neurons_processed.add(succ)
        d._effective_deepness = max_deepness

    @classmethod
    def fill_hash(cls, d : nn.Data):
        d._hash = nn.Hasher.caclulate_hash(d)
