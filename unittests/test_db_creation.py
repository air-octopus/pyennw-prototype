# coding=utf-8

import unittest

import os

from engine_creator import *
from neural_network import NeuralNetwork

class Test_SimpleNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_engine(":memory:", "training-data--unittest-simple.json")

    @classmethod
    def tearDownClass(cls):
        Engine.destroy()

    def test__neural_network_creation(self):
        nn0 = NeuralNetwork()
        nn0_id = nn0.save()

        self.assertEqual(len(nn0.data.neurons), 4)
        self.assertEqual(nn0.data.neurons[0].deepness, 0)
        self.assertEqual(nn0.data.neurons[1].deepness, 0)
        self.assertEqual(nn0.data.neurons[2].deepness, 0)
        self.assertEqual(nn0.data.neurons[3].deepness, 1)
        self.assertEqual(nn0.data.hash, '8f20a03bef0e951e1bfb4fc44cda3062')

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
