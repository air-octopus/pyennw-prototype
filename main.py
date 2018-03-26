# coding=utf-8

import ennwdb
#import transfer_functions as tf

#f = tf.relu

# print("f(-5)", f(-5, ()))
# print("f(-1)", f(-1, ()))
# print("f( 0)", f( 0, ()))
# print("f( 1)", f( 1, ()))
# print("f(17)", f(17, ()))


db = ennwdb.NeuralNetworkDB()
db.open(".temp/temp.db")
