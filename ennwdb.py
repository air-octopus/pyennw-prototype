"""
Операции с базой данных
"""

import  os
import sqlite3 as sql
import util.sqlite_helper

class NeuralNetworkDB:
    def open(self, database_path):
        """
        Открыть существующую базу данных с информацией о нейронных сетях или создать новую.
        :param data_base_path:
        :return:
        """

        self.db = sql.connect(database_path)
        self._create_structure();

    def _create_structure(self):
        c = self.db.cursor()
        c.execute(
            '''
            CREATE TABLE IF NOT EXISTS nn_species_all (
                id          INTEGER PRIMARY KEY,
                parent_id   INTEGER,
                is_alive    INTEGER
            )
            '''
        )
        c.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_nn_species_all_001 ON nn_species_all (
                is_alive
            )
            '''
        )

        c.execute(
            '''
            CREATE TABLE IF NOT EXISTS synapses (
                id              INTEGER PRIMARY KEY,
                species_id      INTEGER,
                neuron_in_id    INTEGER,
                neuron_owner_id INTEGER,
                weight          NUMERIC
            )
            '''
        )
        c.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_synapses_001 ON synapses (
                species_id
            )
            '''
        )


        c.execute(
            '''
            CREATE TABLE IF NOT EXISTS neuron_bodies (
                id                          INTEGER PRIMARY KEY,
                species_id                  INTEGER,
                transfer_function_type      INTEGER,
                transfer_function_params    STRING
            )
            '''
        )
        c.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_neuron_bodies_001 ON neuron_bodies (
                species_id
            )
            '''
        )






