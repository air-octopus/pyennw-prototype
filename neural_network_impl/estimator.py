# coding=utf-8

import neural_network_impl as nn

import collections
import math
import numpy as np
import scipy as sp
import scipy.stats.stats as stat

# from scipy.stats.stats import pearsonr

class Estimator:

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

    @classmethod
    def estimate_adaptability(cls, d : nn.Data, calc, src_arr, desired_arr):
        # result_arr = np.array(calc.multi(src_arr))
        result_arr = calc.multi(src_arr)
        squared_difference = cls._calc_squared_difference_shifted(result_arr, desired_arr)

        response_time = np.argmin(squared_difference)
        quality = squared_difference[response_time]
        adaptability = (response_time + 1) * quality

        return (
            response_time , # response_time
            quality       , # quality
            adaptability    # adaptability
        )

    @classmethod
    def _calc_squared_difference_shifted(cls, a, b):
        """

        :param a:
        :param b:
        :return:
        """

        la = len(a)
        lb = len(b)
        assert la >= lb
        cnt = la - lb + 1

        def sqrdiff(aa, bb):
            aa = np.reshape(aa, (-1))
            bb = np.reshape(bb, (-1))
            cc = aa - bb
            return np.matmul(cc, cc)

        result = np.ndarray(cnt)
        for offset in range(cnt):
                result[offset] = sqrdiff(a[offset:offset + lb], b)

        return result
