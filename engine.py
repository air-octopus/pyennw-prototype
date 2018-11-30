# coding=utf-8

"""
Класс движка предоставляет доступ ко всем компонентам программы
"""

class Engine:

    _instance = None

    @classmethod
    def instance(cls): return cls._instance

    @classmethod
    def db(cls): return cls._instance._db
    @classmethod
    def config(cls): return cls._instance._config
    @classmethod
    def training_data(cls): return cls._instance._training_data

    # @property
    # def db(self): return self._db
    # @property
    # def config(self): return self._config
    # @property
    # def training_data(self): return self._training_data

    def __init__(self):
        if Engine._instance is not None:
            raise Exception("Only one engine is allowed")
