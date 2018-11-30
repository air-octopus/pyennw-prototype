# coding=utf-8

import neural_network_impl as nn
from neural_network_impl.transfer_functions import Type
from engine import Engine

class SaveLoad:

    def save(self, data):
        db = Engine.db()

        # сохраняем id родительской нейросети и получаем (создаем) id текущей
        # todo: реализовать вычисление времени отклика, качества и приспособленности НС
        data.nnid = db.save_species(data.extra_data["parent"],
                                    data.effective_deepness,
                                    data.response_time,
                                    data.resolving_ability,
                                    data.quality,
                                    data.adaptability)

        # todo: идея с массивом nids мне не нра -- лучше записывать идентификаторы непосредственно в нейроны

        # сохраняем параметры нейронов и получаем их идентификаторы
        nids = []
        for o in data.neurons:
            nid = db.save_neuron_body(data.nnid, o.transfer_function_type, json.dumps(o.transfer_function_params), len(o.axon))
            nids.append(nid)
        # ... теперь nids[i] -- идентификатор i-го нейрона

        # сохраняем информацию о синапсах ...
        for o in data.synapses:
            db.save_synapse(data.nnid, nids[o.src], nids[o.own], o.weight)

        # ... входных ...
        for z in zip(data.input_neurons, data.extra_data["input_sids"]):
            db.save_nn_inputs(data.nnid, nids[z[0]], z[1])

        # ... и выходных нейронах
        for z in zip(data.output_neurons, data.extra_data["output_sids"]):
            db.save_nn_outputs(data.nnid, nids[z[0]], z[1])

        db.sqldb.commit()

    def load(self, id):
        """
        загружает данные нейросети с идентификатором id из базы данных
        """
        # todo: to be implemented (загрузка из базы данных)

        # self._temp_data = Builder.TempData()
        self._data = nn.Data()

        self._load_synapses(id)
        self._load_neuron_bodies(id)
        self._load_neuron_inputs(id)
        self._load_neuron_outputs(id)

        data = self._data
        self.__clear()
        return data

    def _load_synapses(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        synapses_data = Engine.db().load_synapses_data(nnid)
        print(len(synapses_data))
        raise Exception('Not implemented')
        pass


    def _load_neuron_bodies(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        raise Exception('Not implemented')
        pass


    def _load_neuron_inputs(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        raise Exception('Not implemented')
        pass


    def _load_neuron_outputs(self, nnid):
        # todo: to be implemented (загрузка из базы данных)
        raise Exception('Not implemented')
        pass
