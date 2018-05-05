import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from datetime import datetime

tfd = tfp.distributions
tfb = tfp.distributions.bijectors

sample_data = np.random.standard_t(4, size=1000)


'''
Attempted implementation of fitting procedure found in SAS documentation
http://support.sas.com/documentation/cdl/en/etsug/66840/HTML/default/viewer.htm#etsug_copula_details06.htm

'''
n_epochs = 10

tf.reset_default_graph()
t_copula = tf.Graph()

with t_copula.as_default():

    input_obs = tf.placeholder(tf.float32)

    dfs = tf.Variable(tf.random_uniform([1, 1]), name='degrees_of_freedom')
    # locs = tf.variable(name='locations')
    # scales = tf.variable(name='scales')

    t_dist = tfd.StudentT(df=dfs, loc=0.0, scale=1.0)

    log_probs = t_dist.log_prob(input_obs, name='log_probs')
    neg_log_like = -1.0*tf.reduce_sum(log_probs)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, name='mle_opt')
    training_op = optimizer.minimize(neg_log_like)

    init = tf.global_variables_initializer()

with tf.Session(graph=t_copula) as sess:

    sess.run(init)

    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={input_obs: sample_data})

        best_dfs = neg_log_like.eval(
            feed_dict={input_obs: sample_data},
            session=sess
        )
        print(best_dfs)


plt.hist(sample_data)
plt.show()
