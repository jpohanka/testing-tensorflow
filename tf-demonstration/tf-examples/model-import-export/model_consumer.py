import tensorflow as tf

path = 'tf-demonstration/tf-examples/model-import-export'
saver = tf.train.import_meta_graph(path+'/saved_model/linear_regression-1900.meta')

with tf.Session() as sess:
    saver.restore(sess, path+'/saved_model/linear_regression-1900')
    x_sample = sess.graph.get_collection("inputs")[0]
    y_calc = sess.graph.get_collection("y_calc")[0]
    
    print(sess.run(y_calc, feed_dict = {
     x_sample:[[0.0], [1.0], [2.0], [3.0]],
    }))