import tensorflow as tf

ooo = tf.zeros(2)

in_01 = tf.constant(ooo)
v_01 = tf.Variable(1)
#v_02 = v_01.
out_01 = v_01 * in_01[1]

sess = tf.Session()
print(sess.run(out_01))