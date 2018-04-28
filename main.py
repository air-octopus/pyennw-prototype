# coding=utf-8

import os

# import from enn_engine
from engine import *
#import ennwdb
#import training_data
#import transfer_functions as tf

#f = tf.relu

# print("f(-5)", f(-5, ()))
# print("f(-1)", f(-1, ()))
# print("f( 0)", f( 0, ()))
# print("f( 1)", f( 1, ()))
# print("f(17)", f(17, ()))


if (os.access(".temp/temp.db", os.F_OK)):
    os.remove(".temp/temp.db")

engine = Engine(".temp/temp.db", "data/training-data.json")

nn = engine.neural_network_loader.load_neural_network(0)

nn.save(engine)

# db = ennwdb.NeuralNetworkDB()
# db.open(".temp/temp.db")
#
# j = training_data.TrainingData(db)
# j.load_file(".temp/training-data.json")
