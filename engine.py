# coding=utf-8
"""
Класс движка предоставляет доступ ко всем компонентам программы
"""

import ennwdb
import training_data as td
import neural_network_loader as nnl

class Engine:
    def __init__(self, db_path: str, training_data_path: str):

        self.db = ennwdb.NeuralNetworkDB()
        self.db.open(db_path)

        self.training_data = td.TrainingData(self.db)
        self.training_data.load_file(training_data_path)

        self.neural_network_loader = nnl.NeuralNetworkLoader(self)
