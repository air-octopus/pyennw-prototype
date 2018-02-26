"""
Класс, описывающий целиком всю нейронную сеть и методы взаимодействия с ней
"""

import copy

class NeuralNetwork:
    def __init__(self):
        # составляющие нейронной сети
        self._synapses = []
        self._neurons  = []


        pass

    def clone(self):
        return copy.copy(self)



