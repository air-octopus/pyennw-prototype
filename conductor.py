# coding=utf-8

"""
Управление очередями мутаций
"""

from engine import Engine

class Conductor:
    def __init__(self):
        # очереди нейросетей. Содержат кортежи с данными вида:
        #   (id, adaptability)
        # adaptability=0 означает наиболее приспособленную сеть
        self._nn_queue_active = Engine.db().get_alive_species()
        self._nn_queue_pending = []
        if len(self._nn_queue_active) == 0:
            self._nn_queue_active = [(0, 0)]

    def next(self):
        if len(self._nn_queue_active) == 0:
            self._nn_queue_pending, self._nn_queue_active = self._nn_queue_active, self._nn_queue_pending
            self._nn_queue_active.sort(key=lambda x: x[1])
            max_len = Engine.config().alive_neural_network_queue_len
            if len(self._nn_queue_active) > max_len:
                Engine.db().extinct_species(self._nn_queue_active[max_len:])
                self._nn_queue_active = self._nn_queue_active[:max_len]

        o = self._nn_queue_active[-1]
        self._nn_queue_active.pop()
        self.add(o[0], o[1])
        return o[0]


    def add(self, id, adaptability):
        if id:
            self._nn_queue_pending.append((id, adaptability))

    def get_best(self):
        key = lambda o: o[1]
        m1 = min(self._nn_queue_active, key=key) if len(self._nn_queue_active) > 0 else None
        m2 = min(self._nn_queue_pending, key=key) if len(self._nn_queue_pending) > 0 else None
        return min(m1, m2, key=key) if m1 and m2 else m1 if m1 else m2


