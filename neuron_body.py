"""
Класс, представляющий методы управления телом нейрона
"""
class neuron_body:
    def __init__(self):
        """
        Инициализация пустого объекта
        """
        # массив синапсов (взвешенных входных данных)
        self._synapses = ()
        # аксон (очередь выходных данных)
        # длина очереди -- не менее одного элемента
        self._axon = [0]

    def calculate(self):
        if len(self._axon) > 1:
            self._axon[1:] = self._axon[0:-1]

    def _transfer(self, v):



