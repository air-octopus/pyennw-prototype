# coding=utf-8

import unittest

from engine_creator import *
from neural_network import NeuralNetwork

nn_json = """
{
    "effective_deepness": 1,
    "response_time": 0,
    "resolving_ability": -1,
    "quality": null,
    "adaptability": null,
    "extra_data": {
        "parent": null,
        "input_sids": [ "input_01", "input_02", "input_03" ],
        "output_sids": [ "unittest_simple" ]
    },
    "neurons": [
        { "id": 1, "axon_len": 1, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 2, "axon_len": 1, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 3, "axon_len": 1, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 4, "axon_len": 2, "bias": 0, "transfer_function_type": 2, "transfer_function_params": [] },
        { "id": 5, "axon_len": 2, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 6, "axon_len": 2, "bias": 0, "transfer_function_type": 2, "transfer_function_params": [] }
    ],
    "synapses": [
        { "src": 1, "own": 5, "weight": 1 },
        { "src": 2, "own": 4, "weight": 1 },
        { "src": 3, "own": 4, "weight": 1 },
        { "src": 4, "own": 5, "weight": 1 },
        { "src": 4, "own": 6, "weight": 1 },
        { "src": 5, "own": 6, "weight": 1 }
    ],
    "input_neurons": [ 1, 2, 3 ],
    "output_neurons": [ 6 ]
}
"""

class Test_NeuralNetworkCalculations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_engine(":memory:", "training-data--unittest-simple.json")

    @classmethod
    def tearDownClass(cls):
        Engine.destroy()

    def test__neural_network_calculation_multistep(self):
        nn0 = NeuralNetwork.from_json(nn_json)

        calc = nn0.calculator

        r = calc.multi([
              (1, 1, 1)
            , (1, 1, 1)
            , (1, 1, 1)
            , (1, 1, 1)
            , (1, 1, 1)
            , (1, 1, 1)
        ])
        r = [ list(o) for o in r ]

        # должно быть sigmoid([[0.0], [3.0], [5.0], [5.0], [5.0], [5.0]])
        self.assertEqual(r, [[0.0], [3.0], [5.0], [5.0], [5.0], [5.0]])

        # todo: добавить тесты на использование transfer functions
        # для сигмоиды будет [[0.5], [0.95257413], [0.9933072], [0.9933072], [0.9933072], [0.9933072]]

        pass

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
