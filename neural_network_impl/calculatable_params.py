# coding=utf-8

import neural_network_impl as nn

import collections

class CalculatableParams:

    # class NeuronPresentation:
    #     pass

    @classmethod
    def fill_deepness(cls, d : nn.Data):
        neuron_ind_to_successors_inds = collections.defaultdict(list)
        for s in d.synapses:
            neuron_ind_to_successors_inds[s.src].append(s.own)

        for i in d.input_neurons:
            d.neurons[i].effective_deepness = 0

        neurons_processed = set(d.input_neurons)
        processing_queue = d.input_neurons.copy()
        max_deepness = 0
        while len(processing_queue) > 0:
            i = processing_queue.pop(0)
            deepness = d.neurons[i].effective_deepness
            for i_next in neuron_ind_to_successors_inds[i]:
                if i_next not in neurons_processed:
                    d.neurons[i_next].effective_deepness = deepness + 1
                    if max_deepness <= deepness: max_deepness = deepness + 1
                    processing_queue.append(i_next)
                    neurons_processed.add(i_next)
        d._effective_deepness = max_deepness
