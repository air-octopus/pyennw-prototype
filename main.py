# coding=utf-8

import os

from engine_creator import *
#from engine import Engine
#import ennwdb
#import training_data
#import transfer_functions as tf

#import neural_network_impl.network_builder

from neural_network import NeuralNetwork

#f = tf.relu

# print("f(-5)", f(-5, ()))
# print("f(-1)", f(-1, ()))
# print("f( 0)", f( 0, ()))
# print("f( 1)", f( 1, ()))
# print("f(17)", f(17, ()))


#import neural_network_impl as nn

# from neural_network_impl.data import Data
# from neural_network_impl.network_builder import Builder
# from neural_network import NeuralNetwork

#d = Data()
# nn = NeuralNetwork(0)



if (os.access(".temp/temp.db", os.F_OK)):
    os.remove(".temp/temp.db")

create_engine(".temp/temp.db", "data/training-data.json")
# engine = Engine(".temp/temp.db", "data/training-data.json")
# engine = Engine.instance()

nn = NeuralNetwork(0)
nn.save()

# builder = nn.Builder()
# d = builder.build_protozoan()

# nn = engine.neural_network_loader.load_neural_network(0)
#
# nn.save(engine)

# db = ennwdb.NeuralNetworkDB()
# db.open(".temp/temp.db")
#
# j = training_data.TrainingData(db)
# j.load_file(".temp/training-data.json")

pass
