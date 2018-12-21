# coding=utf-8

"""
Вычисления на нейросети.
Режим обучения

в режиме обучения:
    * изначально задается необходимое количество итераций в одном батче
    * нейросеть разворачивается сразу на все итерации
    * все итерации в одном батче вычисляются как единое целое
    * результатом является массив ответов нейросети
    * функция потерь учитывает сразу все ответы нейросети
    * веса задаются как tf.Variable
    * начальные данные аксонов инициализируются нулями, и в дальнейшем аккумулируют в себе все предыдущие вычисления
"""

import tensorflow as tf
import neural_network_impl as nn
from engine import Engine

class Trainer(nn.CalculatorBase):
    def __init__(self, d : nn.Data):
        super().__init__(d)

        # Массивы тензоров векторизованного представления нейросети.
        #
        # В режиме тренировки нейросети входные данные бьются на батчи по несколько записей.
        # Каждой записи соответствует одна итерация работы нейросети и, соответственно,
        # один элемент в этих массивах

        # массив tf-плейсхолдеров для входных данных.
        self._in  = None
        # массив данных в аксонах
        self._a   = None
        # массив выходных данных
        self._out = None

        # тензор весов.
        # В режиме тренировки представляет собой экземпляр tf.Variable()
        self._w   = None

        # функция потерь, которая будет минимизироваться в процессе тренировки
        self._loss = None

    def init(self, iterations_count):
        assert iterations_count > 0

        tf.reset_default_graph()

        # все компоненты представлены списками, т.к. для обучения НС итерации вычислений
        # должны быть развернуты на один батч. Соответственно каждой итерации соответствует один элемент в списке.
        self._in      = []
        self._a       = []
        self._out     = []
        self._desired = []
        self._w = tf.Variable([synapse.weight for synapse in self._data.synapses], dtype=tf.float32)

        for i in range(iterations_count):
            self._add_iteration()

        self._loss = None
        #skip = (int)(self._data._response_time)
        skip = 0 # MK_DEBUG
        for out in self._out:
            desired = tf.placeholder(dtype=tf.float32, shape=out.shape)
            self._desired.append(desired)
            if skip > 0:
                skip -= 1
                continue
            dloss = tf.reduce_mean(tf.squared_difference(out, desired))
            self._loss = dloss if self._loss is None else self._loss + dloss

    def training(self, steps_count):
        sess = tf.Session()

        # todo: удалить все ненужное

        # todo: learning_rate должен быть или в настройках или вычисляться адаптивно
        #optim = tf.train.GradientDescentOptimizer(learning_rate=0.0025)  # Оптимизатор
        optim = tf.train.AdamOptimizer(learning_rate=0.25)  # Оптимизатор

        grads_and_vars = optim.compute_gradients(self._loss)

        train_step = optim.minimize(self._loss)

        loss_val = -1
        sess.run(tf.global_variables_initializer())
        for batch in self._training_set_batch(steps_count):
            feed_data = {}
            for tf_in, tf_desired, training_set in zip(self._in, self._desired, batch):
                feed_data[tf_in] = training_set.data_in
                feed_data[tf_desired] = training_set.data_out
            result = sess.run([train_step, grads_and_vars, self._loss, self._out], feed_dict=feed_data)
            loss_val = result[2]
            # all_data = [
            #     self._loss,
            #     self._w,
            #     self._in,
            #     self._a,
            #     self._out,
            #     self._desired
            # ]
            #
            # all_data = sess.run(all_data, feed_dict=feed_data)
            pass

        weights = sess.run(self._w)
        for synapse, w in zip(self._data.synapses, weights):
            synapse.weight = w

        # todo: вынести отдельно весь код оценки нейросети
        # сейчас для простоты используем вычисленное значение функции потерь,
        # однако в дальнейшем надо будет оценивать и время реакции и разрешающую способность и проч
        self._data._quality = loss_val

    def _add_iteration(self):
        """
        В режиме тренировки нейросеть разворачивается сразу на несколько итераций.
        """
        # текущие выходные значения рабочих нейронов
        a2 = self._a[-1] if len(self._a) > 0 else tf.constant([0] * self._a_len, dtype=tf.float32)

        receptors, a1, indicators = self._build_iteration_body(a2)

        self._in .append(receptors )
        self._a  .append(a1        )
        self._out.append(indicators)

    # def _training_set_batch(self, steps_count):
    #     batch = None
    #     batch_len = len(self._in)
    #     step = 0
    #     for ts in Engine.training_data().training_set_loopped():
    #         if batch is None:
    #             batch_len = len(self._in)
    #             batch = []
    #             #batch = [[], []]
    #         batch.append(ts)
    #         # batch[0].append(ts.data_in)
    #         # batch[1].append(ts.data_out)
    #         if len(batch) >= batch_len:
    #             yield batch
    #             step += 1
    #             if step >= steps_count:
    #                 return
    #             batch = None
    def _training_set_batch(self, steps_count):
        for step, batch in zip(range(steps_count), Engine.training_data().training_set_batch_gen(len(self._in))):
            yield batch
            # if batch is None:
            #     batch_len = len(self._in)
            #     batch = []
            #     #batch = [[], []]
            # batch.append(ts)
            # # batch[0].append(ts.data_in)
            # # batch[1].append(ts.data_out)
            # if len(batch) >= batch_len:
            #     yield batch
            #     step += 1
            #     if step >= steps_count:
            #         return
            #     batch = None
