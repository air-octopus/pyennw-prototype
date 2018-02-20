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

        pass

    def _create_structure(self):
        




