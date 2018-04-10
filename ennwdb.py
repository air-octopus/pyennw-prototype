# coding=utf-8

"""
Операции с базой данных
"""

#import os
import sqlite3 as sql
#import util.sqlite_helper

class NeuralNetworkDB:
    def open(self, database_path):
        """
        Открыть существующую базу данных с информацией о нейронных сетях или создать новую.
        :param data_base_path:
        :return:
        """

        self.db = sql.connect(database_path)
        self._create_structure()
        self._cursor = self.db.cursor()

    def add_inputs(self, input_sid):
        self._cursor.execute("INSERT OR IGNORE INTO inputs_all(sid) VALUES('" + input_sid + "')")
        return self._cursor.lastrowid

    def add_outputs(self, output_sid):
        self._cursor.execute("INSERT OR IGNORE INTO outputs_all(sid) VALUES('" + output_sid + "')")
        return self._cursor.lastrowid

    def get_synapses_data(self, species_id):
        '''
        load synapses data from the database
        :param species_id:
        :return: list of tuples (weight, neuron_in_id, neuron_owner_id)
        '''
        return self._cursor.execute("SELECT weight, neuron_in_id, neuron_owner_id FROM synapses WHERE species_id = " + str(species_id))

    def _create_structure(self):
        # query_create_parameters     = ''''''
        # query_create_inputs_all     = ''''''
        # query_create_outputs_all    = ''''''
        # query_create_nn_species_all = ''''''
        # query_create_synapses       = ''''''
        # query_create_neuron_bodies  = ''''''
        # query_create_nn_input       = ''''''
        # query_create_nn_output      = ''''''

        query_create_parameters = '''     
            CREATE TABLE IF NOT EXISTS parameters (
                sid         TEXT UNIQUE,
                value       REAL
            )
        '''
        query_create_inputs_all = '''     
            CREATE TABLE IF NOT EXISTS inputs_all (
                id          INTEGER PRIMARY KEY,
                sid         TEXT UNIQUE
            )
        '''
        query_create_outputs_all = '''    
            CREATE TABLE IF NOT EXISTS outputs_all (
                id          INTEGER PRIMARY KEY,
                sid         TEXT UNIQUE
            )
        '''
        query_create_nn_species_all = ''' 
            CREATE TABLE IF NOT EXISTS nn_species_all (
                id          INTEGER PRIMARY KEY,
                parent_id   INTEGER,
                is_alive    INTEGER
            )
        '''
        query_create_synapses = '''       
            CREATE TABLE IF NOT EXISTS synapses (
                id              INTEGER PRIMARY KEY,
                species_id      INTEGER,
                neuron_in_id    INTEGER,
                neuron_owner_id INTEGER,
                weight          REAL
            )
        '''
        query_create_neuron_bodies = '''  
            CREATE TABLE IF NOT EXISTS neuron_bodies (
                id                          INTEGER PRIMARY KEY,
                species_id                  INTEGER,
                transfer_function_type      INTEGER,
                transfer_function_params    TEXT,
                axon_length                 INTEGER CHECK(axon_length > 0)
            )
        '''
        query_create_nn_input = '''       
            CREATE TABLE IF NOT EXISTS nn_input (
                neuron_id                   INTEGER,
                input_id                    INTEGER,
                species_id                  INTEGER
            )
        '''
        query_create_nn_output = '''      
            CREATE TABLE IF NOT EXISTS nn_output (
                neuron_id                   INTEGER,
                output_id                   INTEGER,
                species_id                  INTEGER
            )
        '''

        c = self.db.cursor()

        c.execute(query_create_parameters    )
        c.execute(query_create_inputs_all    )
        c.execute(query_create_outputs_all   )
        c.execute(query_create_nn_species_all)
        c.execute(query_create_synapses      )
        c.execute(query_create_neuron_bodies )
        c.execute(query_create_nn_input      )
        c.execute(query_create_nn_output     )

#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_species_all_001 ON nn_species_all (is_alive)   ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_synapses_001       ON synapses       (species_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_neuron_bodies_001  ON neuron_bodies  (species_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_input_001       ON nn_input       (synapse_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_output_001      ON nn_output      (neuron_id)  ')

        c.execute("INSERT OR IGNORE INTO parameters(sid, value) VALUES('alive_neural_network_queue_len'              , 100  )")
        c.execute("INSERT OR IGNORE INTO parameters(sid, value) VALUES('mutator_neuron_deleting_probability_factor'  , 0.01 )")
        c.execute("INSERT OR IGNORE INTO parameters(sid, value) VALUES('mutator_synapse_deleting_probability_factor' , 0.05 )")
        c.execute("INSERT OR IGNORE INTO parameters(sid, value) VALUES('mutator_neuron_adding_probability_factor'    , 0.01 )")
        c.execute("INSERT OR IGNORE INTO parameters(sid, value) VALUES('mutator_synapse_adding_probability_factor'   , 0.05 )")

        self.db.commit()
