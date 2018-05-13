import tensorflow as tf

saver = tf.train.import_meta_graph('saved_model/linear_regression.ckpt-1900.meta')

with tf.Session() as sess:
    saver.restore(sess, 'saved_model/linear_regression.ckpt-1900')
    x_sample = sess.graph.get_collection("inputs")[0]
    y_calc = sess.graph.get_collection("y_calc")[0]
    
    print(sess.run(y_calc, feed_dict = {
     x_sample:[[0.0], [1.0], [2.0], [3.0]],
    }))