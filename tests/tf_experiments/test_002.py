import tensorflow as tf
import time

# Срезы тензоров

# t = tf.constant([[[1, 1, 1], [2, 2, 2]],
#                  [[3, 3, 3], [4, 4, 4]],
#                  [[5, 5, 5], [6, 6, 6]]])
#
# tt0 = tf.reshape(tf.slice(t, [0, 0, 0], (1, 2, 3)), (2, 3))
# tt2 = tf.reshape(tf.slice(t, [2, 0, 0], (1, 2, 3)), (2, 3))
# tt = tf.stack([tt0, tt2])
# print(sess.run(tt))

val = [9, 8, 7, 6, 5, 4, 3, 2, 1]
sz = len(val)

for ttt in range(0, 100):
    g = tf.Graph()
    with g.as_default():

        sess = tf.Session()

        #zero = tf.constant([0])
        o0 = tf.constant(val)
        #o1, o2 = tf.split(o0, (sz-1, 1), 0)
        #o2[0] = tf.constant([0])
        #o1 = tf.split(o0, (sz-1), 0)

        o3 = o0
        print("start: ", sess.run(o3)[sz-1])

        t0 = time.time()
        for i in range(1, 100000):
            t1 = time.time()
            o3 = tf.manip.roll(o3, shift=1, axis=0)
            val = sess.run(o3)
            dt = (time.time()-t1)*1000
            if (i % 20 == 0):
                print("[%.3f]> " % (time.time()-t0), " dur=%.3f ms; " % dt,  i, ": ", val[sz-1])
            if (i % 100 == 0):
                break
                # sess.close()
                # sess = tf.Session()
                # # tf.Graph().as_default()
                # o3 = tf.constant(val)




exit(0)

#o3 = tf.manip.roll(o3, shift=1, axis=0)
#
#print(sess.run(o3)[sz-1])
#o3 = tf.manip.roll(o3, shift=1, axis=0)
#
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o1), 0)
#
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o3), 0)
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o3), 0)
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o3), 0)
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o3), 0)
#print(sess.run(o3)[sz-1])
#
#o3 = tf.concat((zero, o3), 0)
#print(sess.run(o3)[sz-1])



in_01 = tf.constant(val)
v_01 = tf.Variable(1)
#v_02 = v_01.
out_01 = v_01 * in_01[1]

#sess = tf.Session()
print(sess.run(out_01))
