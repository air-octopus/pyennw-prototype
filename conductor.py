# coding=utf-8

"""
Управление очередями мутаций
"""

from engine import Engine

class Conductor:
    def __init__(self, engine : Engine):
        self._engine = engine
        # очереди нейросетей. Содержат кортежи с данными вида:
        #   (id, adaptability)
        # adaptability=0 означает наименее приспособленную сеть
        self._nn_queue_active = engine.db.get_alive_species()
        self._nn_queue_pending = []
        if len(self._nn_queue_active) == 0:
            self._nn_queue_active = [(0, 0)]

    def next(self):
        if len(self._nn_queue_active) == 0:
            self._nn_queue_pending, self._nn_queue_active = self._nn_queue_active, self._nn_queue_pending
            self._nn_queue_active.sort(key=lambda x: x[1], reverse=True)
            max_len = self._engine.config.alive_neural_network_queue_len()
            # todo: при удалении сетей из очереди их надо помечать неактивными в базе данных
            if len(self._nn_queue_active) > max_len:
                self._nn_queue_active = self._nn_queue_active[:max_len]

        o = self._nn_queue_active[-1]
        self._nn_queue_active.pop()
        self._nn_queue_pending.append(o)
        return o[0]


    def add(self, id, adaptability):
        self._nn_queue_pending.append((id, adaptability))
