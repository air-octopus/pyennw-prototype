# coding=utf-8

class Synapse:
    """
    Класс-структура, описывающий один синапс
    """

    def __init__(self, neuron_src, neuron_own, weight):
        """
        :param neuron_src: индекс нейрона-источника данных
        :param neuron_own: индекс нейрона-владельца синапса
        :param weight: вес синапса
        """
        self.src = neuron_src
        self.own = neuron_own
        self.weight = weight
