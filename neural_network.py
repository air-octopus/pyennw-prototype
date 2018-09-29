# coding=utf-8

"""
Методы описания нейронной сети и работы с ней в процессе обучения и/или использования
"""

import copy
import json

# from engine import *

class NeuralNetwork:
    """
    Класс, описывающий нейронную сеть.
    Структура нейросети:
        * _data
          массив массивов имеющий смысл массива нейронов и входных данных.
          Каждый нейрон описывается массивом значений, представляющих собой данные аксона.
          Нулевой элемент этого массива -- аккумулятор для текущих вычислений на нейроне.
          Последний элемент -- текущее значение нейрона.
          Самый простейший аксон состоит не менее чем из двух элементов
          Входные данные представляют собой массив из одного элемента
        * _synapses
          массив триплетов, описывающих синапсы. Элементы триплетов имеют следующий смысл:
            * вес синаптической связи
            * индекс нейрона, являющегося входным для данного синапса
            * индекс нейрона, являющегося выходным для синапса
    Note: _synapses определяет конфигурацию нейросети и состояние ее обученности.
          _data определяет оперативные данные.
          Зануление всех элементов _data сбрасывает нейросеть в начальное состояние.
          Зануление _synapses уничтожит нейросеть
    Нейросеть работает тактами. На каждом такте выполняются следующие действия:
        * заполняются входные данные
        * данные аксонов смещаются на один шаг (по направлению к хвосту)
        * начала аксонов, представляющие собой аккумуляторы для вычислений на нейроне зануляются
        * для всех синапсов выполняются вычисления. Результат вычислений аккумулируется в начале соответствующего аксона
        * для всех нейронов выполняется вычисление значения передаточной функции. Источником и результатом является аккумулятор нейрона
        * заполняются выходные данные
    """


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


    class Neuron:
        """
        Класс-структура, описывающий один нейрон
        """
        def __init__(self, axon, transfer_function, transfer_function_params):
            """
            :param axon: массив-очередь данных нейрона (аксон)
            :param transfer_function: передаточная функция нейрона
            :param transfer_function_params: параметры передаточной функции
            """
            self.axon = axon
            self.transfer_function = transfer_function
            self.transfer_function_params = transfer_function_params
            self.effective_deepness = 0


    def __init__(self, neural_network_loader):      # neural_network_loader : nnl.NeuralNetworkLoader
        # текущая нейросеть пока что существует только в памяти, поэтому имеет невалидный идентификатор
        # Валидный идентификатор появится только после сохранения нейросети в БД
        self.nnid = 0
        # составляющие нейронной сети
        self._input_neurons             = neural_network_loader.input_neurons()
        self._output_neurons            = neural_network_loader.output_neurons()
        self._synapses                  = neural_network_loader.synapses()
        self._neurons                   = neural_network_loader.neurons()
        self._effective_deepness        = neural_network_loader.effective_deepness()
        self._extra_data                = neural_network_loader.extra_data()

        self.response_time              = -1
        self.resolving_ability          = -1
        self.quality                    = -1
        self.adaptability               = -1


    def save(self, engine_):        # engine_ : Engine
        db = engine_.db

        # сохраняем id родительской нейросети и получаем (создаем) id текущей
        # todo: реализовать вычисление времени отклика, качества и приспособленности НС
        self.nnid = db.save_species(self._extra_data["parent"],
                                    self._effective_deepness,
                                    self.response_time,
                                    self.resolving_ability,
                                    self.quality,
                                    self.adaptability)

        # сохраняем параметры нейронов и получаем их идентификаторы
        nids = []
        for o in self._neurons:
            nid = db.save_neuron_body(self.nnid, o.transfer_function.type.value, json.dumps(o.transfer_function_params), len(o.axon))
            nids.append(nid)
        # ... теперь nids[i] -- идентификатор i-го нейрона

        # сохраняем информацию о синапсах ...
        for o in self._synapses:
            db.save_synapse(self.nnid, nids[o.src], nids[o.own], o.weight)


        # ... входных ...
        for z in zip(self._input_neurons, self._extra_data["input_sids"]):
            db.save_nn_inputs(self.nnid, nids[z[0]], z[1])

        # ... и выходных нейронах
        for z in zip(self._output_neurons, self._extra_data["output_sids"]):
            db.save_nn_outputs(self.nnid, nids[z[0]], z[1])

        db.sqldb.commit()


    def load_inputs(self, inputs):
        """
        Загрузка входных данных в нейросеть.
        """
        for input_ind, val in enumerate(inputs):
            self._neurons[self._input_neurons[input_ind]].axon[0] = val


    def get_outputs(self, ):
        """
        Выгрузка выходных данных
        """
        return list(self._neurons[output_ind].axon[-1] for output_ind in self._output_neurons)


    def do_iteration(self, ):
        # shortcats
        neurons     = self._neurons
        synapses    = self._synapses

        # прокручиваем аксоны
        for n in neurons:
            if (len(n.axon) > 1): # исключаем рецепторы (у которых длина аксона == 1)
                n.axon = [0] + n.axon[0:-1]

        # выполняем действия над синапсами
        for synapse in synapses:
            neurons[synapse.own].axon[0] += neurons[synapse.src].axon[-1] * neurons[synapse.own].weight

        # применяем передаточную функцию
        for n in neurons:
            if (len(n.axon) > 1): # исключаем рецепторы (у которых длина аксона == 1)
                n.axon[0] = n.transfer_function.func(n.axon[0], n.transfer_function_params)


    def reset(self, ):
        """
        Сбросить состояние нейросети в исходное.
        Состояние -- это те данные, которые могут меняться в процессе работы нейросети
        (в отличие от данных, описывающих структуру нейросети).
        Таковыми являются данные, сохраненные в аксонах
        """
        for n in self._neurons:
            n.axon = [0] * len(n.axon)


    def clone(self):
        """
        Создать копию нейросети. Все данные (и состояние нейросети и ее структура) будут скопированы.
        """
        return copy.deepcopy(self)


    def clone_clean(self):
        """
        Создать нейросеть с такой же структурой, но с начальным состоянием
        """
        nn = self.clone()
        nn.reset()
        return nn


    def clone_state(self):
        """
        Сохранить состояние нейросети
        :return: состояние нейросети (на данный момент состояние содержится в массиве аксонов)
        """
        [copy.deepcopy(n.axon) for n in self._neurons]


    def restore_state(self, state):
        """
        Восстановить состояние нейросети, сохраненное с помощью метода clone_state()
        :param state: состояние НС, полученное ранее с помощью метода clone_state()
        """
        for z in zip(self._neurons, state):
            z[0].axon = z[1]
