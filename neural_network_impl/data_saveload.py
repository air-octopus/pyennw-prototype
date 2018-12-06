# coding=utf-8

import neural_network_impl as nn
from neural_network_impl.transfer_functions import Type
from engine import Engine

import json

class SaveLoad:

    def save(self, data):
        db = Engine.db()

        # сохраняем id родительской нейросети и получаем (создаем) id текущей
        # todo: реализовать вычисление времени отклика, качества и приспособленности НС
        data._id = db.save_species(data.extra_data["parent"],
                                    data.effective_deepness,
                                    data.response_time,
                                    data.resolving_ability,
                                    data.quality,
                                    data.adaptability)

        neurons = data.neurons
        # сохраняем параметры нейронов и получаем их идентификаторы
        for o in neurons:
            o.id = db.save_neuron_body(data.id, o.transfer_function_type, json.dumps(o.transfer_function_params), len(o.axon))

        # сохраняем информацию о синапсах ...
        for o in data.synapses:
            db.save_synapse(data.id, neurons[o.src].id, neurons[o.own].id, o.weight)

        # ... входных ...
        for ind, sid in zip(data.input_neurons, data.extra_data["input_sids"]):
            db.save_nn_inputs(data.id, neurons[ind].id, sid)

        # ... и выходных нейронах
        for ind, sid in zip(data.output_neurons, data.extra_data["output_sids"]):
            db.save_nn_outputs(data.id, neurons[ind].id, sid)

        db.sqldb.commit()

        return data.id

    def load(self, id):
        """
        загружает данные нейросети с идентификатором id из базы данных
        """
        d = self.__data = nn.Data()

        (
              d._extra_data["parent_id"]
            , d._effective_deepness
            , d._response_time
            , d._resolving_ability
            , d._quality
            , d._adaptability
        ) = Engine.db().load_species(id)

        self._load_neuron_bodies(id)
        self._load_neuron_inputs(id)
        self._load_neuron_outputs(id)
        self._load_synapses(id)

        # todo: Нужно выполнять корректировку рецепторов/индикторов под текущие входные и выходные данные (возможно придется пересортировать)
        # По сути это означает, что основным открытым интерфейсом становится nn.Builder. т.е. стек вызовов будет такой:
        #       NeuralNetwork.__init__(...)
        #       --> Builder.load(...)
        #           --> SaveLoad.load(...)
        #

        data = self.__data
        self.__data = None
        return data

    def _load_neuron_bodies(self, nnid):
        neurons_data = Engine.db().load_neurons_data(nnid)
        self.__data._neurons = [
            nn.Neuron(id, [0] * axon_len, tf_type, json.loads(tf_params))
            for id, tf_type, tf_params, axon_len
            in neurons_data
        ]
        self.__data._map_neuron_id2ind = None

    def _load_neuron_inputs(self, nnid):
        map_neuron_id2ind = self.__data.map_neuron_id2ind
        inputs = Engine.db().load_nn_inputs(nnid)
        self.__data._input_neurons            = [ map_neuron_id2ind[id] for id, input_sid in inputs ]
        self.__data._extra_data["input_sids"] = [ input_sid             for id, input_sid in inputs ]

    def _load_neuron_outputs(self, nnid):
        map_neuron_id2ind = self.__data.map_neuron_id2ind
        outputs = Engine.db().load_nn_outputs(nnid)
        self.__data._output_neurons            = [ map_neuron_id2ind[id] for id, output_sid in outputs ]
        self.__data._extra_data["output_sids"] = [ output_sid            for id, output_sid in outputs ]

    def _load_synapses(self, nnid):
        map_neuron_id2ind = self.__data.map_neuron_id2ind
        synapses_data = Engine.db().load_synapses_data(nnid)
        self.__data._synapses = [
            nn.Synapse(map_neuron_id2ind[neuron_in_id], map_neuron_id2ind[neuron_owner_id], weight)
            for weight, neuron_in_id, neuron_owner_id
            in synapses_data
        ]