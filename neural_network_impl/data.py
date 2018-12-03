# coding=utf-8

import copy

# from neural_network_impl.synapse import Synapse
# from neural_network_impl.neuron import Neuron

class Data:
    """
    Данные, которыми оперирует нейросеть.
    Экземпляр этого класа находится в объекте нейросети, но также используется для передачи данных нейросети.
    (например при загрузке данных из БД или при мутации).

    Данные нейросети разделяются на
        * идентификатор (выделяется в самостоятельную категорию в виду своей важности)
        * конфигурация
        * оперативные данные
        * вспомогательные данные

    Идентификатор определяет нейросеть в БД

    Конфигурация определяется:
        * массивом нейронов с указанием их свойств -- тип и параметры передаточной функции, длина аксона.
          Стоит заметить, что сами значения, записанные в аксонах сюда не входят -- они являются оперативными данными
        * связями между нейронами. Задается с помощью массива синапсов. Каждый синапс -- одна связь
        * весами связей

    Оперативные данные определяются:
        * значениями в аксонах

    вспомогательные данные:
        * результаты оценки нейросети (качество, время реакции....)
        * предок нейросети
        * типы входных и выходных данных
        * ....
    """

    @property
    def id                  (self): return self._id
    @property
    def input_neurons       (self): return self._input_neurons
    @property
    def output_neurons      (self): return self._output_neurons
    @property
    def synapses            (self): return self._synapses
    @property
    def neurons             (self): return self._neurons
    @property
    def effective_deepness  (self): return self._effective_deepness
    @property
    def extra_data          (self): return self._extra_data
    @property
    def response_time       (self): return self._response_time
    @property
    def resolving_ability   (self): return self._resolving_ability
    @property
    def quality             (self): return self._quality
    @property
    def adaptability        (self): return self._adaptability

    @property
    def map_neuron_id2ind   (self):
        if self._map_neuron_id2ind  is None:
            self.rebuild_indices()
        return self._map_neuron_id2ind

    def __init__(self):

        # по-умолчанию предполагаем, что текущая нейросеть пока что существует только в памяти, поэтому имеет невалидный идентификатор
        # Валидный идентификатор появится только после сохранения нейросети в БД
        self._id = 0

        #############################
        # основные данные нейронной сети, определяющие ее конфигурацию и свойства.

        # массив индексов нейронов-рецепторов
        self._input_neurons             = []
        # массив индексов нейронов-индикаторов
        self._output_neurons            = []
        # массив синапсов (объектов класса Synapse)
        self._synapses                  = []
        # массив всех нейронов (объектов класса Neuron)
        self._neurons                   = []

        #############################
        # вспомогательные данные

        # эффективная глубина нейросети
        self._effective_deepness        = -1
        # дополнительные данные, которые надо будет записать в БД
        self._extra_data                = {
            "parent"        : 0 ,
            "input_sids"    : [],
            "output_sids"   : [],
        }

        # время реакции
        self._response_time             = -1
        # разрешающая временная способность НС
        self._resolving_ability         = -1
        # качество НС
        self._quality                   = -1
        # приспособляемость НС
        self._adaptability              = -1

        #############################
        # индексы

        # Отображение id нейрона в его индекс
        self._map_neuron_id2ind = None

        # Отображение sid входа в индекс нейрона-рецептора
        # self._map_input_sid2ind = dict()
        # Отображение sid выхода в индекс нейрона-индикатора
        # self._map_output_sid2ind = dict()

    def rebuild_indices(self):
        self._map_neuron_id2ind = dict()
        for ind, neuron in enumerate(self._neurons):
            self._map_neuron_id2ind[neuron.id] = ind

    def reset(self):
        """
        Сбросить оперативные данные (т.е. привести состояние нейросети в исходное)
        """
        for n in self.neurons:
            n.axon = [0] * len(n.axon)

    def clone(self):
        """
        Создать копию данных нейросети. Все данные (и состояние нейросети и ее структура) будут скопированы.
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
        return [copy.deepcopy(n.axon) for n in self.data.neurons]

    def restore_state(self, state):
        """
        Восстановить состояние нейросети, сохраненное с помощью метода clone_state()
        :param state: состояние НС, полученное ранее с помощью метода clone_state()
        """
        for neuron, axon in zip(self.data.neurons, state):
            neuron.axon = axon

