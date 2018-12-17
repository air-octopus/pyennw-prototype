# coding=utf-8

from engine import Engine

import neural_network_impl as nn

# import collections
# import hashlib
# import json

class Mutator:

    def __init__(self, data : nn.Data):
        self._data = None

    def mutate(self, data : nn.Data):
        pass

    def _gen_ids(self):
        """
        В рабочем режиме нейроны и синапсы нейросети (а значит и вся геометрия сети) оперделяются их индексами.
        Однако в процессе мутирования геометрия меняется и индексы не подходят.
        Везде в пределах данного файла будем оперировать не индексами, а "идентификаторами"
        :return:
        """

    def _remove_neurons(self):
        pass

    pass
