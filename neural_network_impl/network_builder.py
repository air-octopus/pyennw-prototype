# coding=utf-8

import neural_network_impl as nn
from neural_network_impl.transfer_functions import Type

class Builder:

    class TempData:
        def __init__(self, engine):
            self._map_neuron_id2ind  = dict()  # Отображение id нейрона в его индекс
            self._map_input_sid2ind  = dict()  # Отображение sid входа в индекс нейрона-рецептора
            self._map_output_sid2ind = dict()  # Отображение sid выхода в индекс нейрона-индикатора
            self.engine = engine

        @property
        def input_neurons(self):
            return list(self._map_input_sid2ind[sid] for sid in self.engine.training_data.inputs)

        @property
        def output_neurons(self):
            return list(self._map_output_sid2ind[sid] for sid in self.engine.training_data.outputs)

    @property
    def engine(self): return self._engine
    @property
    def data(self): return self._data

    def __init__(self, engine):
        self._engine = engine

    def __clear(self):
        self._data = None
        self._temp_data = None

    def build_protozoan(self):
        """
        Создает данные для самой простейшей базовой нейросети.
        По сути эта нейросеть состоит только из нейронов-рецепторов, нейронов-индикаторов
        и связей между ними по типу все-со-всеми
        """
        self._temp_data = Builder.TempData(self.engine)
        self._data = nn.Data()
        self._adapt_for_inputs_and_outputs()
        self._calc_effective_deepness()
        data = self._data
        self.__clear()
        return data

    def __add_receptor(self, input_data_name):
        new_neuron_ind = len(self.data.neurons)
        new_neuron_tf = Type.relu
        new_neuron_tf_params = ()
        new_neuron = nn.Neuron([0], new_neuron_tf, new_neuron_tf_params)

        self._temp_data._map_input_sid2ind [input_data_name] = new_neuron_ind
        self.data.neurons.append(new_neuron)

    def __add_indicator(self, output_data_name):
        new_neuron_ind = len(self.data.neurons)
        new_neuron_tf = Type.linear
        new_neuron_tf_params = ()
        new_neuron = nn.Neuron([0, 0], new_neuron_tf, new_neuron_tf_params)
        # В настоящее время при создании нейрона-индикатора он присоединяется напрямую к рецептору
        # (у которого глубина по-определению равна нулю),
        # поэтому в данном случае глубина индикатора будет 1
        # А вообще... надо todo: добавить функцию перевычисления эффективной глубины нейронов и нейросети
        new_neuron.effective_deepness = 1

        self._temp_data._map_output_sid2ind [output_data_name] = new_neuron_ind
        self.data.neurons.append(new_neuron)

    def _adapt_for_inputs_and_outputs(self):
        """
        Модифицируем нейронную сеть так, что бы она могла принимать на вход имеющиеся
        и выдавать требуемые данные.
        В результате работы функции будут добавлены:
            * недостающие нейроны-рецепторы и -индикаторы
            * связи между добавленными рецепторами и индикаторами по типу все-со-всеми
        """
        current_inputs_set  = set(self.engine.training_data.inputs)
        current_outputs_set = set(self.engine.training_data.outputs)
        have_inputs_set     = set(self._temp_data._map_input_sid2ind.keys())
        have_outputs_set    = set(self._temp_data._map_output_sid2ind.keys())

        inputs_to_add     = current_inputs_set .difference(   have_inputs_set )
        inputs_to_delete  = have_inputs_set    .difference(current_inputs_set )
        outputs_to_add    = current_outputs_set.difference(   have_outputs_set)
        outputs_to_delete = have_outputs_set   .difference(current_outputs_set)

        # На каждый новый вход и на каждый новый выход добавляем по одному нейрону
        # между ними делаем связки все-со-всеми
        # todo: думаю в дальнейшем, когда тестовые данные будут усложняться эта логика изменится

        for data_name in inputs_to_add:  self.__add_receptor (data_name)
        for data_name in outputs_to_add: self.__add_indicator(data_name)

        # рассматриваем связи все-со-всеми
        for src in (self._temp_data._map_input_sid2ind[src_sid] for src_sid in inputs_to_add):
            for own in (self._temp_data._map_output_sid2ind[own_sid] for own_sid in outputs_to_add):
                # добавляем синаптическую связь
                self._data._synapses.append(nn.Synapse(src, own, 1)) # todo: вынести значение веса по-умолчанию в настройки

        # todo: добавить реализацию для удаления рецепторов и/или индикаторов

        self.data._input_neurons  = self._temp_data.input_neurons
        self.data._output_neurons = self._temp_data.output_neurons

    def _calc_effective_deepness(self):
        for n in self.data.neurons:
            n.effective_deepness = -1
        for i in self.data.input_neurons:
            self._data._neurons[i].effective_deepness = 0

        stop = False

        while not stop:
            stop = True
            replaces_count = 0
            for synapse in self.data.synapses:
                src_effective_deepness = self.data.neurons[synapse.src].effective_deepness
                own_effective_deepness = self.data.neurons[synapse.own].effective_deepness
                if own_effective_deepness < 0:
                    if src_effective_deepness >= 0:
                        self.data.neurons[synapse.own].effective_deepness = src_effective_deepness + 1
                        replaces_count += 1
                    else:
                        stop = False

            if replaces_count == 0:
                assert False    # видимо в графе имеются повисшие куски
                # break

        self.data._effective_deepness = -1
        for n in self._data._neurons:
            if n.effective_deepness > self.data.effective_deepness:
                self.data._effective_deepness = n.effective_deepness

