from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import distributions as tfcd

g = tf.Graph()

"""
The idea comes from Lemma 1 from the paper "Some Characterizations of the Multivariate t Distribution":

https://ac.els-cdn.com/0047259X72900218/1-s2.0-0047259X72900218-main.pdf?_tid=9035319c-e165-4b9a-8851-ec6837fcfe0e&acdnat=1527542834_a6193c0ddc8296cfcf2bc38c73536890
"""

with g.as_default():
    
    # Create a vectorized t-distribution.
    studentT = tfcd.StudentT(
        df=2.1,
        loc=[[0.0,0.0]],
        scale=[[1.0,1.0]]
    )
    
    # Scale matrix for the multivariate t-distribution.
    scale_matrix = [[2.0, 0.5], [0.5, 2.0]]
    
    # Compute the Cholesky decomposition of the scale matrix.
    tril = tf.cholesky([scale_matrix])[0]
    
    # In this name scope we create affine transform of the vectorized t-distribution
    # which will result in a 2-D t-distribution with a prescribed scale matrix.
    with tf.name_scope("multi_studentT"):
    
        # Create the multivariate t-distribution via an affine transform with
        # a lower-trianguler matrix.
        multi_studentT = tfcd.TransformedDistribution(
          distribution=studentT,
          bijector=tfcd.bijectors.Affine(scale_tril=tril),
          name="MultiStudentT",
        )
        
        # Derive some quantities from the multivariate t-distribution.
        multi_studentT_prob = multi_studentT.prob([[0.1,0.01]])
        multi_studentT_log_prob = multi_studentT.log_prob([[0.1,0.01]])
        multi_studentT_sample = multi_studentT.sample(10)
    

with tf.Session(graph=g) as sess:
    # Save the model graph.
    writer = tf.summary.FileWriter(
        graph=g,
        logdir="summary_files/mtd",
    )
    
    # Close the FileWriter and write the data to disk.
    writer.close()
    
    print("Compute the probability:")
    print(sess.run(multi_studentT_prob))
    
    print("Compute the log-probability:")
    print(sess.run(multi_studentT_log_prob))
    
    print("--------------")
    
    print("Draw a sample set:")
    print(sess.run(multi_studentT_sample))
