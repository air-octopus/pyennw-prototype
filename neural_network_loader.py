# coding=utf-8

"""
Загрузка параметров нейронной сети из базы данных и создание соответствующего экземпляра нейронной сети
"""

import neural_network as nn

class NeuralNetworkLoader:
    def __init__(self, enndb):
        self._enndb = enndb


    def load_neural_network(self, nnid):
        if (nnid == 0):
            return self._build_protozoan()

        self._load_synapses(nnid)
        self._load_neuron_bodies(nnid)
        self._load_neuron_inputs(nnid)
        self._load_neuron_outputs(nnid)
        return self._build_neural_network()


    def _load_synapses(self, nnid):
        # todo: to be implemented
        synapses_data = self._enndb.get_synapses_data(nnid)
        print(len(synapses_data))
        pass


    def _load_neuron_bodies(self, nnid):
        # todo: to be implemented
        pass


    def _load_neuron_inputs(self, nnid):
        # todo: to be implemented
        pass


    def _load_neuron_outputs(self, nnid):
        # todo: to be implemented
        pass


    def _adapt_for_inputs(self):
        # todo: to be implemented
        pass


    def _adapt_for_outputs(self):
        # todo: to be implemented
        pass


    def _build_protozoan(self):
        # todo: to be implemented
        '''
        Построить самую простейшую нейросеть.
        Метод применяется в самом начале работы программы, когда в базе данных нет ни одной нейросети
        Простейшая нейросеть не содержит ничего :) ни нейронов ни синапсов. Создается только общая структура и внутренние переменные
        Поэтому после такого "создания" необходимо запускать адаптацию нейросети к входным и выходным данным
        '''
        pass


    def _build_neural_network(self):
        # todo: to be implemented
        pass



