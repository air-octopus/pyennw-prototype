# coding=utf-8

import json
import random

from ennwdb import *


class TrainingData:
    """
    Класс, управляющий исходными данными для обучения нейросети
    Имеет следующие открытые атрибуты:
        self.input_sid_to_ind:  - отображение sid входного параметра в его индекс
        self.output_sid_to_ind: - отображение sid выходного параметра в его индекс
        self.inputs:            - массив sid'ов входных параметров
        self.outputs:           - массив sid'ов выходных параметров
        self.training_set:      -
        self.testing_set:       -
        self.combo_set:         -
    """

    class Row:
        def __init__(self, data_in, data_out):
            self.data_in = data_in
            self.data_out = data_out

    def __init__(self, db: NeuralNetworkDB = None, json_str = None):
        self._db = db
        if json_str is None:
            json_str = '''
                {
                    "training_set_name": "null",
                    "inputs": [],
                    "outputs": [],
                    "combo_set_ratio": 0,
                    "combo_set": [],
                    "training_set": [],
                    "testing_set": []
                }
            '''
        self.load_str(json_str)

    def load_file(self, file_name):
        f = open(file=file_name)
        s = f.read()
        self.load_str(s)

    def load_str(self, str):
        j = json.loads(str)
        self.json_all = j
        self.input_sid_to_ind = {}
        self.output_sid_to_ind = {}
        inputs = j['inputs']
        outputs = j['outputs']
        self.inputs            = inputs
        self.outputs           = outputs
        self._training_set     = j["training_set"]
        self._testing_set      = j["testing_set"]
        combo_set              = j["combo_set"]
        combo_set_ratio        = j["combo_set_ratio"]

        for ind, sid in enumerate(inputs):
            self.input_sid_to_ind[sid] = ind
        for ind, sid in enumerate(outputs):
            self.output_sid_to_ind[sid] = ind

        random.shuffle(combo_set)
        combo_set_training_len = int(combo_set_ratio * len(combo_set))
        self._training_set.extend(combo_set[:combo_set_training_len])
        self._testing_set.extend(combo_set[combo_set_training_len:])

        if self._db is not None:
            for i in range(0, len(inputs)):
                self._db.add_inputs(inputs[i])
            for i in range(0, len(outputs)):
                self._db.add_outputs(outputs[i])
            self._db.sqldb.commit()

    def training_set_size(self):
        return len(self._training_set)

    def training_set_loopped(self):
        while True:
            for ts in self._training_set:
                yield TrainingData.Row(ts["in"], ts["out"])

    def training_set_portion(self, count):
        result = []
        for row in self.training_set_loopped():
            result.append(row)
            if len(result) >= count:
                break
        return result

    def training_set(self):
        for ts in self._training_set:
            yield TrainingData.Row(ts["in"], ts["out"])

    def testing_set_size(self):
        return len(self._testing_set)

    def testing_set(self):
        for ts in self._testing_set:
            yield TrainingData.Row(ts["in"], ts["out"])
