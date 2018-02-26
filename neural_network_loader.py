"""
Загрузка параметров нейронной сети из базы данных и создание соответствующего экземпляра нейронной сети
"""

import neural_network as nn

class NeuralNetworkLoader:
    def __init__(self, nndb):
        self._nndb = nndb

    def load_nn(self, nnid): # todo
        pass