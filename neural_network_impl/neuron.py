# coding=utf-8

class Neuron:
    """
    Класс-структура, описывающий один нейрон
    """

    def __init__(self, axon, transfer_function_type, transfer_function_params):
        """
        :param axon: массив-очередь данных нейрона (аксон)
        :param transfer_function: передаточная функция нейрона
        :param transfer_function_params: параметры передаточной функции
        """
        self.axon = axon
        self.transfer_function_type = transfer_function_type
        self.transfer_function_params = transfer_function_params
        self.effective_deepness = 0

        self.__yyy = 42


