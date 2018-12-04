# coding=utf-8

import tensorflow as tf

# Перестановка элементов тензора

ab = (1, 3, 4, 4)
bc = (0, 0, 0, 1, 3, 3)

# Так нельзя!
# ttt = tf.constant([
#     [1, 2, 3, 4, 5],
#     [6, 7],
#     [8, 9, 10]
# ])

a = tf.constant([7, 6, 5, 4, 3, 2, 1])
b = tf.gather(a, ab)
c = tf.gather(b, bc)

print(tf.Session().run(b))
print(tf.Session().run(c))

tf.zeros()

# результат:
#   [6 4 3 3]
#   [6 6 6 4 3 3]
