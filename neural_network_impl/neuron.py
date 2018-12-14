# coding=utf-8

class Neuron:
    """
    Класс-структура, описывающий один нейрон
    """

    @property
    def is_receptor(self):
        return True if len(self.axon) == 1 else False

    def __init__(self, axon, transfer_function_type, transfer_function_params):
        """
        :param axon: массив-очередь вещественных чисел -- данных нейрона (аксон)
        :param transfer_function_type: тип передаточной функции нейрона
        :param transfer_function_params: параметры передаточной функции
        """
        self.axon = axon
        self.transfer_function_type = transfer_function_type
        self.transfer_function_params = transfer_function_params
        self.effective_deepness = 0

