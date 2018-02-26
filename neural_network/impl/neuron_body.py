"""
Класс, представляющий методы управления телом нейрона
"""

import transfer_functions as tf

class NeuronBody:
    def __init__(self):
        """
        Инициализация пустого объекта
        """

        # массив синапсов (взвешенных входных данных)
        self._synapses = ()

        # аксон (очередь выходных данных)
        # длина очереди -- не менее одного элемента
        self._axon = [0]

        # Передаточная функция
        self._trasfer_func = tf.linear

    def calculate(self):
        if len(self._axon) > 1:
            self._axon[1:] = self._axon[0:-1]

        self._axon[0] = sum(synapse.value() for synapse in self._synapses)

    def value(self):
        """
        :return: текущее значение, вычисленное в данном нейроне
        """
        return self._axon[-1]
