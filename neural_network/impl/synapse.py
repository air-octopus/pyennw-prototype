# coding=utf-8

"""
Класс для описания синапса
"""


class Synapse:
    def __init__(self, neuron_in):
        self.weight = 1
        self.neuron_in = neuron_in

    def value(self):
        """
        :return: взвешенное значение синапса
        """
        return self.neuron_in.value() * self.weight
