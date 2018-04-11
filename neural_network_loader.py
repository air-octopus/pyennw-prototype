# coding=utf-8
"""
Загрузка параметров нейронной сети из базы данных и создание соответствующего экземпляра нейронной сети
"""

import engine
import neural_network as nn
import transfer_functions as tf

class NeuralNetworkLoader:
    def __init__(self, engine_):
        self._engine = engine_
        self._db = self._engine.db
        self._training_data = self._engine.training_data


    def load_neural_network(self, nnid):
        self._extra_data_parent = nnid

        if nnid == 0:
            return self._build_protozoan()

        self._load_synapses(nnid)
        self._load_neuron_bodies(nnid)
        self._load_neuron_inputs(nnid)
        self._load_neuron_outputs(nnid)
        return self._build_neural_network()


    # Доступ к внутренним данным
    def input_neurons             (self): return self._input_neurons
    def output_neurons            (self): return self._output_neurons
    def synapse_weights           (self): return self._synapse_weights
    def synapse_sources           (self): return self._synapse_sources
    def synapse_owners            (self): return self._synapse_owners
    def axons                     (self): return self._axons
    def transfer_functions        (self): return self._transfer_functions
    def transfer_functions_params (self): return self._transfer_functions_params
    def map_neuron_id2ind         (self): return self._map_neuron_id2ind
    def extra_data                (self):
        extra_data_value = dict()
        extra_data_value["parent"    ] = self._extra_data_parent
        extra_data_value["input_sids"] =  self._training_data.inputs.copy()
        extra_data_value["output_sids"] =  self._training_data.outputs.copy()
        extra_data_value["transfer_functions_types"] =  [tf.type for tf in self._transfer_functions]
        return extra_data_value


    def _load_synapses(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        synapses_data = self._db.load_synapses_data(nnid)
        print(len(synapses_data))
        pass


    def _load_neuron_bodies(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        pass


    def _load_neuron_inputs(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        pass


    def _load_neuron_outputs(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        pass

    def _add_receptor(self, input_data_name):
        new_neuron_ind = len(self._axons)
        new_neuron_tf = tf.TransferFunction(tf.Type.relu)
        new_neuron_tf_params = ()

        #self._map_neuron_id2ind         =
        self._map_input_sid2ind          [input_data_name] = new_neuron_ind
        self._axons                     .append([0])
        self._transfer_functions        .append(new_neuron_tf)
        self._transfer_functions_params .append(new_neuron_tf_params)

    def _add_indicator(self, output_data_name):
        new_neuron_ind = len(self._axons)
        new_neuron_tf = tf.TransferFunction(tf.Type.linear)
        new_neuron_tf_params = ()

        #self._map_neuron_id2ind         =
        self._map_output_sid2ind         [output_data_name] = new_neuron_ind
        self._axons                     .append([0, 0])
        self._transfer_functions        .append(new_neuron_tf)
        self._transfer_functions_params .append(new_neuron_tf_params)


    def _adapt_for_inputs_and_outputs(self):
        """
        Модифицируем нейронную сеть так, что бы она могла принимать на вход имеющиеся
        и выдавать требуемые данные
        """
        current_inputs_set  = set(self._training_data.inputs)
        current_outputs_set = set(self._training_data.outputs)
        have_inputs_set     = set(self._map_input_sid2ind.keys())
        have_outputs_set    = set(self._map_output_sid2ind.keys())

        inputs_to_add     = current_inputs_set.difference(have_inputs_set     )
        inputs_to_delete  = have_inputs_set   .difference(current_inputs_set  )
        outputs_to_add    = current_outputs_set.difference(have_outputs_set   )
        outputs_to_delete = have_outputs_set   .difference(current_outputs_set)

        # На каждый новый вход и на каждый новый выход добавляем по одному нейрону
        # между ними делаем связки все-со-всеми
        # todo: думаю в дальнейшем, когда тестовые данные будут усложняться эта логика изменится

        for data_name in inputs_to_add:  self._add_receptor (data_name)
        for data_name in outputs_to_add: self._add_indicator(data_name)

        # рассматриваем связи все-со-всеми
        for src in (self._map_input_sid2ind[src_sid] for src_sid in inputs_to_add):
            for own in (self._map_output_sid2ind[own_sid] for own_sid in outputs_to_add):
                # добавляем синаптическую связь
                self._synapse_weights.append(1)     # todo: вынести значение веса по-умолчанию в настройки
                self._synapse_sources.append(src)
                self._synapse_owners .append(own)

        # todo: добавить реализацию для удаления рецепторов и/или индикаторов

        self._input_neurons  = list(self._map_input_sid2ind [sid] for sid in self._training_data.inputs )
        self._output_neurons = list(self._map_output_sid2ind[sid] for sid in self._training_data.outputs)


    def _build_protozoan(self):
        """
        Построить самую простейшую нейросеть.
        Метод применяется в самом начале работы программы, когда в базе данных нет ни одной нейросети
        Простейшая нейросеть не содержит ничего :) ни нейронов ни синапсов. Создается только общая структура и внутренние переменные
        Поэтому после такого "создания" необходимо запускать адаптацию нейросети к входным и выходным данным
        """
        self._synapse_weights           = []
        self._synapse_sources           = []
        self._synapse_owners            = []
        self._axons                     = []
        self._transfer_functions        = []
        self._transfer_functions_params = []
        self._map_neuron_id2ind         = dict()
        self._map_input_sid2ind         = dict()    # Отображение sid входа в индекс нейрона-рецептора
        self._map_output_sid2ind        = dict()    # Отображение sid выхода в индекс нейрона-индикатора
        self._adapt_for_inputs_and_outputs()
        return self._build_neural_network()


    def _build_neural_network(self):
        return nn.NeuralNetwork(self)



