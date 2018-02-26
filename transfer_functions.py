"""
Различные виды передаточных (aka активационных) функций
Для всех функций:
    x: значение аргумента передаточной функции
    p: кортеж из нуля, одного или нескольких элементов с параметрами передаточной функции
"""

import math
import enum


def linear(x, p):
    """
    линеййная передаточная функция.
    :param p: должен быть пустым
    """
    return x
def linear1(x) : return linear(x, ())


def relu(x, p):
    """
    линейная ректификация (rectified linear unit)
    :param p: должен быть пустым
    """
    return 0 if x < 0 else x;
def relu1(x) : return relu(x, ())

def logistic(x, p):
    """
    Логистическая функция или сигмоида
    :param p[0]: параметр логистической функции
                p[0] = 0..infty
                0 -- логистическая функция вырождается в прямую со значением 0.5
                infty -- логистическая функция вырождается в пороговую
    """
    return 1.0/(1 + math.exp(-p[0]*x))

def step(x, p):
    """
    Пороговая функция
    :param x: значение аргумента
    :param p[0]: сдвиг функции по оси x
    """
    return 0 if x < p[0] else 1


class Type: #(enum.Enum):
    invalid  = 0
    linear   = 1
    relu     = 2
    logistic = 3
    step     = 4

    def __init__(self, type):
        self.type = type

    def func(self):
        if self.type == Type.linear:
            return linear
        elif self.type == Type.relu:
            return relu
        elif self.type == Type.logistic:
            return logistic
        elif self.type == Type.step:
            return step
        else:
            raise RuntimeError("Unsupported transfer function type")
