# coding=utf-8

"""
параметры программы
"""
#from engine import Engine


class Config:
    #def __init__(self, engine: Engine):
    def __init__(self, engine):
        self.engine = engine
        self.config = engine.db.load_config()

    @property
    def alive_neural_network_queue_len             (self): return self.config['alive_neural_network_queue_len'             ]
    @property
    def mutator_neuron_deleting_probability_factor (self): return self.config['mutator_neuron_deleting_probability_factor' ]
    @property
    def mutator_synapse_deleting_probability_factor(self): return self.config['mutator_synapse_deleting_probability_factor']
    @property
    def mutator_neuron_adding_probability_factor   (self): return self.config['mutator_neuron_adding_probability_factor'   ]
    @property
    def mutator_synapse_adding_probability_factor  (self): return self.config['mutator_synapse_adding_probability_factor'  ]
