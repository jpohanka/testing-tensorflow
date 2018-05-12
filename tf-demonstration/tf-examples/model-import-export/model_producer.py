from __future__ import print_function

import tensorflow as tf
import numpy as np

learning_rate = 0.8
learning_epochs = 2000

path = 'tf-demonstration/tf-examples/model-import-export'

g = tf.Graph()

# ------------------------------------------------------------------------------
# Define the model graph.
# ------------------------------------------------------------------------------

with g.as_default():
    with tf.name_scope("inputs"):
        # Input data placeholders.
        x_sample = tf.placeholder(name="x_sample", shape=[None, None], dtype=tf.float64)
        y_sample = tf.placeholder(name="y_sample", shape=[None, None], dtype=tf.float64)
        tf.add_to_collection('inputs', x_sample)
        tf.add_to_collection('inputs', y_sample)
        
    # Define the model variables.
    with tf.variable_scope("model_variables"):
        slope = tf.get_variable(
            name="slope", 
            dtype=tf.float64,
            initializer=tf.constant(1.0, dtype=tf.float64)
        )
        intercept = tf.get_variable(
            name="intercept", 
            dtype=tf.float64,
            initializer=tf.constant(1.0, dtype=tf.float64)
        )
        
        # Added summary reports
        tf.summary.scalar('slope', slope)
        tf.summary.scalar('intercept', intercept)
        tf.summary.histogram('slope', slope)
        tf.summary.histogram('intercept', intercept)
        
    # Define the linear regression model.
    with tf.name_scope("linear_regression"):
        y_calc = slope * x_sample + intercept
        tf.add_to_collection('y_calc', y_calc)
        
    # Define the optimizer for model training.
    with tf.name_scope("optimizer"): 
        loss =  tf.reduce_sum((y_calc - y_sample)**2)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        #
        tf.summary.scalar('loss', slope)
        tf.summary.histogram('loss', loss)
            
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

# ------------------------------------------------------------------------------
# Generate input data
# ------------------------------------------------------------------------------

# Model coefficients, sample size.
sample_size = 100
slope_data = 1.5
intercept_data = -3.0

# Define the residual error.
err_stddev = 0.005
err = np.random.randn(sample_size) * err_stddev

# Define the sample input variable.
x_sample_data = np.arange(sample_size).reshape(1, sample_size) * 1.0

# Define the sample output variable.
y_sample_data = slope_data * x_sample_data + intercept_data + err

# ------------------------------------------------------------------------------
# Set up the Saver instance
# ------------------------------------------------------------------------------

# Define the Saver class instance for persisting the model.
saver = tf.train.Saver(var_list=[slope, intercept])

# ------------------------------------------------------------------------------        
# Run the training phase.
# ------------------------------------------------------------------------------

with tf.Session(graph=g) as sess:
    # Create a summary writer.
    writer = tf.summary.FileWriter(path+"/summaries/linear_regression", graph=sess.graph)
    
    sess.run(init)
    for epoch in range(learning_epochs):
        # Evaluate a list of nodes. By evaluating the optimizer node, we variables
        # 'slope' and 'intercept' will be ipdated in each session run.
        _, loss_val, slope_val, intercept_val, summary = sess.run(
            [
                optimizer, 
                loss, 
                slope, 
                intercept, 
                merged
            ], 
            feed_dict = {
                x_sample: x_sample_data,
                y_sample: y_sample_data
            }
        )
        # Write the summary reports for a given training epoch to the summary writer.
        writer.add_summary(summary, epoch)
        
        # After every 100 epochs, persist the estimated values of the model variables.
        if epoch % 100 == 0:
            print("epoch: %s, loss: %s, slope: %s, intercept: %s" % (epoch, loss_val, slope_val, intercept_val))
            saver.save(sess, global_step=epoch, save_path=path+'/saved_model/linear_regression')
        
