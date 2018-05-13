import tensorflow as tf

# Import the model TF graph via a Saver class.
saver = tf.train.import_meta_graph('saved_model/linear_regression.ckpt-1900.meta')

with tf.Session() as sess:
    
    # Restore the model.
    saver.restore(sess, 'saved_model/linear_regression.ckpt-1900')
    
    # Get references to the input variable 'x_sample' and output variable
    # 'y_calc' from the TF graph via the 'inputs' collection.
    x_sample = sess.graph.get_collection("inputs")[0]
    y_calc = sess.graph.get_collection("y_calc")[0]
    
    # Run a test batch calculation with few input values.
    print(sess.run(y_calc, feed_dict = {
     x_sample:[[0.0], [1.0], [2.0], [3.0]],
    }))