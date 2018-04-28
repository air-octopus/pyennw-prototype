# coding=utf-8

"""
Основной цикл приложения -- эволюционный процесс
"""

from engine import Engine

class EvolutionProcessor:
    def __init__(self, engine : Engine):
        self.engine = engine
        self.conductor = engine.conductor
        self.neural_network_loader = engine.neural_network_loader


    def start_evolution(self):
        for i in range(0, 10):
            self.step()

    def step(self):
        """
        Элементарная операция процесса эволюции
        * получаем очередной идентификатор вида из очереди эволюции и загружаем соответствующую сеть
        * производим мутацию
        * обучаем и оцениваем нейросеть
        * сохраняем нейросеть в базу и в очередь эволюции
        """
        nnid = self.conductor.next()
        nn = self.neural_network_loader.load_neural_network(nnid)

        # todo: мутация
        # todo: обучение и последующая оценка нейросети

        nn.save(self.engine)
        self.conductor.add(nn.nnid, nn.adaptability)



