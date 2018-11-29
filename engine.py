# coding=utf-8

"""
Класс движка предоставляет доступ ко всем компонентам программы
"""

# from conductor import Conductor
from config import Config
from ennwdb import NeuralNetworkDB
# from evolution_processor import EvolutionProcessor
# from neural_network_loader import NeuralNetworkLoader
from training_data import TrainingData

class Engine:

    _instance = None

    @property
    @classmethod
    def instance(cls): return cls._instance

    def __init__(self, db_path: str, training_data_path: str):
        if Engine.instance is not None:
            raise Exception("Only one engine is allowed")

        self.db = NeuralNetworkDB()
        self.db.open(db_path)

        self.config = Config(self)

        self.training_data = TrainingData(self.db)
        self.training_data.load_file(training_data_path)

        # self.neural_network_loader = NeuralNetworkLoader(self)
        #
        # self.conductor = Conductor(self)
        # self.evolution_processor = EvolutionProcessor(self)

        Engine._instance = self
