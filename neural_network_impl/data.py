# coding=utf-8

from neural_network_impl.synapse import Synapse
from neural_network_impl.neuron import Neuron


class Data:
    """
    Данные, которыми оперирует нейросеть.
    Экземпляр этого класа находится в объекте нейросети, но также используется для передачи данных нейросети.
    (например при загрузке данных из БД или при мутации)
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

    def __init__(self):

        # по-умолчанию предполагаем, что текущая нейросеть пока что существует только в памяти, поэтому имеет невалидный идентификатор
        # Валидный идентификатор появится только после сохранения нейросети в БД
        self._id = 0

        # составляющие нейронной сети

        # массив индексов нейронов-рецепторов
        self._input_neurons             = []
        # массив индексов нейронов-индикаторов
        self._output_neurons            = []
        # массив синапсов (объектов класса Synapse)
        self._synapses                  = []
        # массив всех нейронов (объектов класса Neuron)
        self._neurons                   = []
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
