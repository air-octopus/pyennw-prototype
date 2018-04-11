# coding=utf-8

"""
Методы описания нейронной сети и работы с ней в процессе обучения и/или использования
Note: не приспособлены для выполнения операций мутации нейронных сетей


"""

import engine
import json

import copy
import neural_network_loader as nnl
from builtins import range


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

    def __init__(self, neural_network_loader):      # neural_network_loader : nnl.NeuralNetworkLoader
        # составляющие нейронной сети
        self._input_neurons             = neural_network_loader.input_neurons()
        self._output_neurons            = neural_network_loader.output_neurons()
        self._synapse_weights           = neural_network_loader.synapse_weights()
        self._synapse_sources           = neural_network_loader.synapse_sources()
        self._synapse_owners            = neural_network_loader.synapse_owners()
        self._axons                     = neural_network_loader.axons()
        self._transfer_functions        = neural_network_loader.transfer_functions()
        self._transfer_functions_params = neural_network_loader.transfer_functions_params()
        self._map_neuron_id2ind         = neural_network_loader.map_neuron_id2ind()
        self._extra_data                = neural_network_loader.extra_data()


    def save(self, engine_):        # engine_ : engine.Engine
        assert len(self._axons) ==len(self._transfer_functions)
        assert len(self._axons) ==len(self._transfer_functions_params)

        db = engine_.db

        # сохраняем id родительской нейросети и получаем (создаем) id текущей
        nnid = db.save_species_parent_id(self._extra_data["parent"])

        # сохраняем параметры нейронов и получаем их идентификаторы
        nids = []
        for z in zip(self._axons, self._transfer_functions, self._transfer_functions_params):
            nid = db.save_neuron_body(nnid, z[1].type.value, json.dumps(z[2]), len(z[0]))
            nids.append(nid)
        # ... теперь nids[i] -- идентификатор i-го нейрона

        # сохраняем информацию о синапсах ...
        for z in zip(self._synapse_sources, self._synapse_owners, self._synapse_weights):
            db.save_synapse(nnid, nids[z[0]], nids[z[1]], z[2])

        # ... входных ...
        for z in zip(self._input_neurons, self._extra_data["input_sids"]):
            db.save_nn_inputs(nnid, nids[z[0]], z[1])

        # ... и выходных нейронах
        for z in zip(self._output_neurons, self._extra_data["output_sids"]):
            db.save_nn_outputs(nnid, nids[z[0]], z[1])

        db.sqldb.commit()


    def load_inputs(self, inputs):
        """
        Загрузка входных данных в нейросеть.
        """
        for input_ind, val in enumerate(inputs):
            self._axons[self._input_neurons[input_ind]][0] = val


    def get_outputs(self, ):
        """
        Выгрузка выходных данных
        """
        return list(self._output_neurons[output_ind] for output_ind in self._output_neurons)


    def do_iteration(self, ):
        # shortcats
        axons           = self._axons
        synapse_weights = self._synapse_weights
        synapse_sources = self._synapse_sources
        synapse_owners  = self._synapse_owners

        axons_count = len(axons)
        synspses_count = len(synapse_weights)

        # прокручиваем аксоны
        for i in range(0, axons_count):
            if (len(axons[i]) > 1): # исключаем рецепторы (у которых длина аксона == 1)
                axons[i] = [0] + axons[i][0:-1]

        # выполняем действия над синапсами
        for i in range(0, synspses_count):
            axons[synapse_owners[i]][0] = axons[synapse_owners[i]][0] + axons[synapse_sources[i]][-1] * synapse_weights[i]

        # применяем передаточную функцию
        for i in range(0, axons_count):
            if (len(axons[i]) > 1): # исключаем рецепторы (у которых длина аксона == 1)
                axons[i][0] = self._transfer_functions[i].func()(axons[i][0], self._transfer_functions_params[i])


    def reset(self, ):
        """
        Сбросить состояние нейросети в исходное.
        Состояние -- это те данные, которые могут меняться в процессе работы нейросети
        (в отличие от данных, описывающих структуру нейросети).
        Таковыми являются данные, сохраненные в аксонах
        """
        for i in range(len(self._axons)):
            self._axons[i] = [0] * len(self._axons[i])


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
        return copy.deepcopy(self._axons)


    def restore_state(self, state):
        """
        Восстановить состояние нейросети, сохраненное с помощью метода clone_state()
        :param state: состояние НС, полученное ранее с помощью метода clone_state()
        """
        self._axons = state
