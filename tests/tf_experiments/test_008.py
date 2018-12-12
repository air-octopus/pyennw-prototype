# coding=utf-8

import tensorflow as tf

# Влияние операций на вычисление градиентов

#========================================================================
#

p3 = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
w = tf.Variable([1, 1, 1, 1, 1], dtype=tf.float32)

p4 = p3 * w
#p4 = w

#a1 = tf.Variable([0, 0, 0, 0, 0], dtype=tf.float32, trainable=False)
#a1 = tf.Variable([0, 0, 0], dtype=tf.float32, trainable=False)
#a1 = tf.constant([0, 0, 0], dtype=tf.float32)
#a1 = p4

# НАДО ИСПОЛЬЗОВАТЬ tf.segment_sum
a1 = tf.segment_sum(p4, [0, 0, 0, 1, 2])

# a1 = tf.scatter_nd_add(a1, [[0], [0], [0], [1], [2]], p4)
#a1 = tf.scatter_nd_add(a1, [0, 0, 0, 1, 2], p4)
# a1 = tf.gather(p4, [0, 3, 4])
# a1 += tf.gather(p4, [1, 3, 4])

# a1 = a1 * p4

out = tf.gather(a1, [0, 2])

loss = tf.reduce_mean(out)

#sess = tf.Session()

optim = tf.train.GradientDescentOptimizer(learning_rate=0.25)  # Оптимизатор

grads_and_vars = optim.compute_gradients(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
o = sess.run([grads_and_vars, loss])

pass
