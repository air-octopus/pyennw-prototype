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
    db = NeuralNetworkDB()
    db.open(db_path)

    config = Config(db)

    training_data = TrainingData(db)
    training_data.load_file(training_data_path)

    # engine.neural_network_loader = NeuralNetworkLoader(engine)
    #
    # engine.conductor = Conductor(engine)
    # engine.evolution_processor = EvolutionProcessor(engine)

    engine = Engine()
    engine._db = db
    engine._config = config
    engine._training_data = training_data

    Engine._instance = engine
