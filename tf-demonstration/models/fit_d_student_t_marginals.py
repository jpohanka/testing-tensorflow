'''
Attempted implementation of fitting procedure found in SAS documentation:
http://support.sas.com/documentation/cdl/en/etsug/66840/HTML/default/viewer.htm#etsug_copula_details06.htm
Implementation of Maximum Likelyhood Estimation for 'dim' independent marginals.

Tensorboard: tensorboard --logdir=/Users/peter/Documents/Pythonithub_repos/testing-tensorflow/tf-demonstration/models/logs
'''

import numpy as np
import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = '/Users/peter/Documents/Python/github_repos/testing-tensorflow/tf-demonstration/models/logs/t_marginals_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)


'''
Set Parameters and generate sample data
'''


class Config():

    true_dfs = 4.0
    true_scales = 1.0
    sample_size = 2500
    init_df_params = {'minval': 2.0, 'maxval': 6.0}
    init_scale_params = {'mean': 1.0, 'stddev': 0.5}
    lr = 0.001
    max_epochs = 100000
    eps_param, eps_loss, eps_grad = 1e-10, 1e-10, 1e-10
    random_seed = 0
    dim = 3


config = Config()

np.random.seed(0)
sample_data = np.random.standard_t(df=config.true_dfs, size=(config.sample_size, config.dim))


np.random.seed(config.random_seed)


'''
Graph Construction
'''

tf.reset_default_graph()
t_copula = tf.Graph()

with t_copula.as_default():

    # tensor for data
    input_data = tf.placeholder(shape=[None, config.dim],
                                dtype=tf.float32, name='input_observations')

    with tf.name_scope('trained_parameters'):

        # tensors for parameters
        dfs = tf.Variable(initial_value=tf.random_uniform([config.dim], **config.init_df_params),
                          dtype=tf.float32, name='degrees_of_freedom')
        scales = tf.Variable(initial_value=tf.random_normal([config.dim], **config.init_scale_params),
                             dtype=tf.float32, name='scales')

        with tf.name_scope('variable_logs'):

            df_hist = tf.summary.histogram('dfs_hist', dfs)
            scales_hist = tf.summary.histogram('scales_hist', scales)

            for dims in range(config.dim):
                df_scalar = tf.summary.scalar('dfs_scalar_dim_'+str(dims), dfs[dims])
                scales_scalar = tf.summary.scalar('scales_scalar_dim_'+str(dims), scales[dims])

    with tf.name_scope('mle_target'):

        # loss function
        t_dist = tf.contrib.distributions.StudentT(
            dfs, loc=tf.zeros([1, config.dim]), scale=scales, name='student_t_RV')
        log_prob = t_dist.log_prob(value=input_data)
        neg_log_like = -1.0 * tf.reduce_sum(log_prob, name='log_observations')
        maxl_summary = tf.summary.scalar('maximum_likelihood', neg_log_like)

    with tf.name_scope('optimizer'):

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr, name='optimizer')
        train_op = optimizer.minimize(loss=neg_log_like, name='training_target')

        # gradient
        grad = tf.gradients(neg_log_like, [dfs, scales], name='gradient')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    merged_summary = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


'''
Training Session
'''

with tf.Session(graph=t_copula) as sess:

    sess.run(init)

    epoch = 1
    obs_dfs, obs_scales = sess.run(fetches=[[dfs], [scales]])
    obs_loss = sess.run(fetches=[neg_log_like], feed_dict={input_data: sample_data})
    obs_grad = sess.run(fetches=[grad], feed_dict={input_data: sample_data})

    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={input_data: sample_data})

        # update parameters
        new_dfs, new_scales = sess.run(fetches=[dfs, scales])
        diff_norm = np.linalg.norm(np.subtract([new_dfs, new_scales],
                                               [obs_dfs[-1], obs_scales[-1]]))

        # update loss
        new_loss = sess.run(fetches=neg_log_like, feed_dict={input_data: sample_data})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad = sess.run(fetches=grad, feed_dict={input_data: sample_data})
        grad_norm = np.linalg.norm(new_grad)

        obs_dfs.append(new_dfs)
        obs_scales.append(new_scales)
        obs_loss.append(new_loss)
        obs_grad.append(new_grad)

        summary_str = merged_summary.eval(feed_dict={input_data: sample_data})
        file_writer.add_summary(summary_str, epoch)

        if epoch % 100 == 0:
            print("Epoch", epoch, ": loss_diff =", loss_diff)
            saver_path = saver.save(
                sess, '/Users/peter/Documents/Python/github_repos/testing-tensorflow/tf-demonstration/models/logs/checkpoints/t_marginals.ckpt')

        if diff_norm < config.eps_param:
            print('Parameter convergence in {} iterations!'.format(epoch))
            break

        if loss_diff < config.eps_loss:
            print('Loss function convergence in {} iterations!'.format(epoch))
            break

        if grad_norm < config.eps_grad:
            print('Gradient convergence in {} iterations!'.format(epoch))
            break

        if epoch >= config.max_epochs:
            print('Max number of iterations reached without convergence.')
            break

        epoch += 1

    saver.save(
        sess, '/Users/peter/Documents/Python/github_repos/testing-tensorflow/tf-demonstration/models/logs/checkpoints/t_marginals_final.ckpt')
    file_writer.close()
