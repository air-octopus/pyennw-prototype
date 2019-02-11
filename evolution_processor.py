# coding=utf-8

"""
Основной цикл приложения -- эволюционный процесс
"""

from engine import Engine
from neural_network import NeuralNetwork

class EvolutionProcessor:
    def __init__(self):
        self.conductor = Engine.conductor()


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
        id = self.conductor.next()
        try:
            nn = NeuralNetwork(id)

            nn.mutate()
            nn.train(steps_count=50)
            nn.estimate(steps_count=10)

            id = nn.save()
            self.conductor.add(id, nn.data.adaptability)
        # finally:
        #     pass
        except:
            print("Error occurred for id=%d" % (id))
            return



