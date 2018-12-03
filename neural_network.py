# coding=utf-8

"""
Методы описания нейронной сети и работы с ней в процессе обучения и/или использования
"""

import copy
import json

#from neural_network_impl.data import Data
import neural_network_impl as nn

from engine import Engine

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

    @property
    def data(self):
        return self._data

    def __init__(self, id):      # neural_network_loader : nnl.NeuralNetworkLoader

        if id == 0:
            self._data = nn.Builder().build_protozoan()
        else:
            self._data = nn.SaveLoad().load(id)

        # self._data = nn.Data()

        # self._input_neurons             = []
        # self._output_neurons            = []
        # self._synapses                  = []
        # self._neurons                   = []
        # self._effective_deepness        = -1
        # self._extra_data                = {
        #     "parent"                  : 0 ,
        #     "input_sids"              : [],
        #     "output_sids"             : [],
        # }
        #
        # self.response_time              = -1
        # self.resolving_ability          = -1
        # self.quality                    = -1
        # self.adaptability               = -1


        # # текущая нейросеть пока что существует только в памяти, поэтому имеет невалидный идентификатор
        # # Валидный идентификатор появится только после сохранения нейросети в БД
        # self.nnid = 0
        # # составляющие нейронной сети
        # self._input_neurons             = neural_network_loader.input_neurons()
        # self._output_neurons            = neural_network_loader.output_neurons()
        # self._synapses                  = neural_network_loader.synapses()
        # self._neurons                   = neural_network_loader.neurons()
        # self._effective_deepness        = neural_network_loader.effective_deepness()
        # self._extra_data                = neural_network_loader.extra_data()
        #
        # self.response_time              = -1
        # self.resolving_ability          = -1
        # self.quality                    = -1
        # self.adaptability               = -1


    def save(self):
        return nn.SaveLoad().save(self._data)
        # db = Engine.db()
        #
        # # сохраняем id родительской нейросети и получаем (создаем) id текущей
        # # todo: реализовать вычисление времени отклика, качества и приспособленности НС
        # self.data.nnid = db.save_species(self.data.extra_data["parent"],
        #                             self.data.effective_deepness,
        #                             self.data.response_time,
        #                             self.data.resolving_ability,
        #                             self.data.quality,
        #                             self.data.adaptability)
        #
        # # todo: идея с массивом nids мне не нра -- лучше записывать идентификаторы непосредственно в нейроны
        #
        # # сохраняем параметры нейронов и получаем их идентификаторы
        # nids = []
        # for o in self.data.neurons:
        #     nid = db.save_neuron_body(self.data.nnid, o.transfer_function_type, json.dumps(o.transfer_function_params), len(o.axon))
        #     nids.append(nid)
        # # ... теперь nids[i] -- идентификатор i-го нейрона
        #
        # # сохраняем информацию о синапсах ...
        # for o in self.data.synapses:
        #     db.save_synapse(self.data.nnid, nids[o.src], nids[o.own], o.weight)
        #
        # # ... входных ...
        # for z in zip(self.data.input_neurons, self.data.extra_data["input_sids"]):
        #     db.save_nn_inputs(self.data.nnid, nids[z[0]], z[1])
        #
        # # ... и выходных нейронах
        # for z in zip(self.data.output_neurons, self.data.extra_data["output_sids"]):
        #     db.save_nn_outputs(self.data.nnid, nids[z[0]], z[1])
        #
        # db.sqldb.commit()

    def load_inputs(self, inputs):
        """
        Загрузка входных данных в нейросеть.
        """
        for neuron_ind, val in zip(self.data.input_neurons, inputs):
            self.data.neurons[neuron_ind].axon[0] = val


    def get_outputs(self):
        """
        Выгрузка выходных данных
        """
        return list(self.data.neurons[output_ind].axon[-1] for output_ind in self.data.output_neurons)


    def do_iteration(self, ):
        # todo: реализовать через tensorflow
        pass
        # # shortcats
        # neurons     = self.data.neurons
        # synapses    = self.data.synapses
        #
        # # прокручиваем аксоны
        # for n in neurons:
        #     if (len(n.axon) > 1): # исключаем рецепторы (у которых длина аксона == 1)
        #         n.axon = [0] + n.axon[0:-1]
        #
        # # выполняем действия над синапсами
        # for synapse in synapses:
        #     neurons[synapse.own].axon[0] += neurons[synapse.src].axon[-1] * neurons[synapse.own].weight
        #
        # # применяем передаточную функцию
        # for n in neurons:
        #     if (len(n.axon) > 1): # исключаем рецепторы (у которых длина аксона == 1)
        #         n.axon[0] = n.transfer_function.func(n.axon[0], n.transfer_function_params)


    def reset(self):
        """
        Сбросить состояние нейросети в исходное.
        """
        self.data.reset()

    def clone(self):
        """
        Создать копию данных нейросети
        """
        return self.data.clone()

    def clone_clean(self):
        """
        Создать нейросеть с такой же структурой, но с начальным состоянием
        """
        return self.data.clone_clean()


    def clone_state(self):
        """
        Сохранить состояние нейросети
        :return: состояние нейросети (на данный момент состояние содержится в массиве аксонов)
        """
        return self.data.clone_state()


    def restore_state(self, state):
        """
        Восстановить состояние нейросети, сохраненное с помощью метода clone_state()
        :param state: состояние НС, полученное ранее с помощью метода clone_state()
        """
        self.data.restore_state(state)
