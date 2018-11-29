# coding=utf-8

"""
Создание класса движка.
Выделено в отдельный файл, что бы разрулить циклические ссылки зависимостей модулей
"""

from engine import Engine
# from conductor import Conductor
from config import Config
from ennwdb import NeuralNetworkDB
# from evolution_processor import EvolutionProcessor
# from neural_network_loader import NeuralNetworkLoader
from training_data import TrainingData

def create_engine(db_path: str, training_data_path: str):
    engine = Engine()
    engine._db = NeuralNetworkDB()
    engine.db.open(db_path)

    engine._config = Config(engine)

    engine._training_data = TrainingData(engine.db)
    engine.training_data.load_file(training_data_path)

    # engine.neural_network_loader = NeuralNetworkLoader(engine)
    #
    # engine.conductor = Conductor(engine)
    # engine.evolution_processor = EvolutionProcessor(engine)

    Engine._instance = engine
