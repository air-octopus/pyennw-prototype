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
        "parent_id": null,
        "input_sids": [ "input_01", "input_02", "input_03" ],
        "output_sids": [ "unittest_simple" ]
    },
    "neurons": [
        { "id": 1, "axon_len": 1, "bias": 0, "transfer_function_type": 2, "transfer_function_params": [] },
        { "id": 2, "axon_len": 1, "bias": 0, "transfer_function_type": 2, "transfer_function_params": [] },
        { "id": 3, "axon_len": 1, "bias": 0, "transfer_function_type": 2, "transfer_function_params": [] },
        { "id": 4, "axon_len": 2, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 5, "axon_len": 2, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] },
        { "id": 6, "axon_len": 2, "bias": 0, "transfer_function_type": 1, "transfer_function_params": [] }
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

class Test_NeuralNetworkSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_engine(":memory:", "training-data--unittest-simple.json")

    @classmethod
    def tearDownClass(cls):
        Engine.destroy()

    def test__neural_network_serialization_deserialization(self):
        nn0 = NeuralNetwork.from_json(nn_json)

        self.assertEqual(len(nn0.data.neurons), 6)
        self.assertEqual(len(nn0.data.synapses), 6)
        self.assertEqual(len(nn0.data.input_neurons), 3)
        self.assertEqual(len(nn0.data.output_neurons), 1)

        nn0_json = nn0.data.serialize_json()

        nn1 = NeuralNetwork.from_json(nn0_json)

        self.assertEqual(len(nn1.data.neurons), 6)
        self.assertEqual(len(nn1.data.synapses), 6)
        self.assertEqual(len(nn1.data.input_neurons), 3)
        self.assertEqual(len(nn1.data.output_neurons), 1)

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
