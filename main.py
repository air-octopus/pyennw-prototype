# coding=utf-8

import os

from engine_creator import *
#from engine import Engine
#import ennwdb
#import training_data
#import transfer_functions as tf

import neural_network_impl as nn

from neural_network import NeuralNetwork

# if (os.access(".temp/temp.db", os.F_OK)):
#     os.remove(".temp/temp.db")

create_engine(".temp/temp.db", "data/training-data.json")
# engine = Engine(".temp/temp.db", "data/training-data.json")
# engine = Engine.instance()

td = Engine.training_data()

nn0 = NeuralNetwork()
nn0_gv = nn0.print_gv()
with open(".temp/nn0.gv", "w") as f:
    f.write(nn0_gv)

for i in range(1, 20):
    nn.Mutator.mutate(nn0.data)
    nn_gv = nn0.print_gv()
    with open(".temp/nn%03d.gv" % i, "w") as f:
        f.write(nn_gv)

exit(0)

nn0 = NeuralNetwork(1)

nn.CalculatableParams.fill_deepness(nn0.data)

hsh = nn.Hasher.caclulate_hash(nn0.data)

for synapse in nn0.data.synapses:
    synapse.weight *= 0.7

nn0._data._response_time = 2

#s = nn0.data.serialize_json()

trainer = nn.Trainer(nn0)
trainer.init(iterations_count=7)

trainer.training(100)

# for t in trainer._training_set_batch(2):
#     print(t)

#nnid = nn0.save()

nn0_gv = nn0.print_gv()

with open(".temp/nn0.gv", "w") as f:
    f.write(nn0_gv)

# nn0.data._response_time     = 14.1234567
# nn0.data._resolving_ability = 15.1234567
# nn0.data._quality           = 16.1234567
# nn0.data._adaptability      = 17.1234567
#nnid = nn0.save()

#nn2 = NeuralNetwork(nnid)
#nnid2 = nn2.save()

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
