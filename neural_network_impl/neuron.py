# coding=utf-8

class Neuron:
    """
    Класс-структура, описывающий один нейрон
    """

    def __init__(self, id, axon, transfer_function_type, transfer_function_params):
        """
        :param axon: массив-очередь вещественных чисел -- данных нейрона (аксон)
        :param transfer_function_type: тип передаточной функции нейрона
        :param transfer_function_params: параметры передаточной функции
        """
        self.id = id
        self.axon = axon
        self.transfer_function_type = transfer_function_type
        self.transfer_function_params = transfer_function_params
        self.effective_deepness = 0

