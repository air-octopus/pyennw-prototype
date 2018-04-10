# coding=utf-8

import json

import ennwdb

class TrainingData:

    def __init__(self, enndb: ennwdb.NeuralNetworkDB):
        self._enndb = enndb

    def load_file(self, file_name):
        f = open(file=file_name)
        s = f.read()
        self.load_str(s)

    def load_str(self, str):
        j = json.loads(str)
        input_sid_to_ind = {}
        output_sid_to_ind = {}
        inputs = j['inputs']
        outputs = j['outputs']
        for i in range(0, len(inputs)):
            input_sid_to_ind[inputs[i]] = i
            self._enndb.add_inputs(inputs[i])
        for i in range(0, len(outputs)):
            output_sid_to_ind[outputs[i]] = i
            self._enndb.add_outputs(outputs[i])

        self._enndb.db.commit()

        self.input_sid_to_ind  = input_sid_to_ind
        self.output_sid_to_ind = output_sid_to_ind
        self.inputs            = inputs
        self.outputs           = outputs
        self.training_set      = j["training_set"]
        self.testing_set       = j["testing_set"]
        self.combo_set         = j["combo_set"]
