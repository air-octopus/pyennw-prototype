# coding=utf-8

import tensorflow as tf

# Терминология:
#   * внутренние нейроны -- те, которые участвуют в вычислениях и имеют синапсы
#   * рецепторы -- нейроны без синапсов, в которые записываются входные данные
#   * нейроны -- объединенная группа {рецепторы, внутренние нейроны}

# индексы нейронов для построения массива синапсов
ra2s       = [0, 1, 5, 5, 1, 2, 3, 4]
# индексы внутренних нейронов, являющихся индикаторами
indind     = [2]
# количество синапсов у каждого внутреннего нейрона
s4alens    = [3, 3, 2]
# индексы (внутренних нейронов) для суммирования синапсов и построения нового массива внутренних нейронов
s2a        = [0, 0, 0, 1, 1, 1, 2, 2]

# рецепторы
r = tf.placeholder("float", shape=[3])
#r = tf.constant([1, 2, 3], dtype=tf.float32)
# внутренние нейроны (a -- axon)
a = tf.constant([0, 0, 0], dtype=tf.float32)
# веса синапсов
w = tf.Variable([1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
# смещения (только для внутренних нейронов)
b = tf.Variable([0, 0, 0], dtype=tf.float32)
# # веса синапсов
# w = tf.Variable([1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
# # смещения (только для внутренних нейронов)
# b = tf.Variable([0, 0, 0], dtype=tf.float32)

ra = tf.concat((r, a), 0)
#s  = tf.gather(ra, ra2s)
s  = tf.multiply(tf.gather(ra, ra2s), w)

a  = tf.nn.relu(tf.segment_sum(s, s2a) + b)

indicators  = tf.gather(a, indind)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [1, 2, 3]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [1, 2, 3]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [1, 2, 3]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [0, 0, 0]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [0, 0, 0]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [0, 0, 0]}))

    ra = tf.concat((r, a), 0)
    s = tf.multiply(tf.gather(ra, ra2s), w)
    a = tf.nn.relu(tf.segment_sum(s, s2a) + b)
    indicators = tf.gather(a, indind)

    print( sess.run(a, feed_dict={r: [1, 2, 3]}))
    print( sess.run(indicators, feed_dict={r: [0, 0, 0]}))


#s4a = tf.split(s, s4alens)

