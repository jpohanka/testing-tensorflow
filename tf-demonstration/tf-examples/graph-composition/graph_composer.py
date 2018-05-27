from __future__ import print_function

import tensorflow as tf

g = tf.Graph()

# Here the approach 1 of the graph composition if applied.
with g.as_default():
    sg1_a = tf.placeholder(shape=[], dtype=tf.float64, name="sg1_a")
    sg1_b = tf.placeholder(shape=[], dtype=tf.float64, name="sg1_b")
    sg2_d = tf.placeholder(shape=[], dtype=tf.float64, name="sg2_d")
    
    # Read the first sub-graph's GraphDef proto.
    with open("graph_def_files/subgraph1_graph_def.pb", "rb") as graph1_file:    
        graph1_graph_def = tf.GraphDef()
        graph1_graph_def.ParseFromString(graph1_file.read())
    
    # Get the output from the first sub-graph
    sg1_out_sum, = tf.import_graph_def(
            graph_def=graph1_graph_def,
            name="sg1",
            input_map={
                "a:0": sg1_a,
                "b:0": sg1_b,
            },
            return_elements=["out_sum:0"],
    )

    # Read the second sub-graph's GraphDef proto.
    with open("graph_def_files/subgraph2_graph_def.pb", "rb") as graph2_file:    
        graph2_graph_def = tf.GraphDef()
        graph2_graph_def.ParseFromString(graph2_file.read())
    
    # Get the output from the second sub-graph
    sg2_out_calc, = tf.import_graph_def(
            graph_def=graph2_graph_def,
            input_map={
                "c:0":sg1_out_sum,
                "d:0":sg2_d,
            },
            name="sg2",
            return_elements=["out_calc:0"],
    )
    
# Save the GraphDef proto twice - in binary and in human-readable pbtxt format.
tf.train.write_graph(
    graph_or_graph_def=g,
    logdir="graph_def_files",
    name="graph_composed.pb",
    as_text=False
)
tf.train.write_graph(
    graph_or_graph_def=g,
    logdir="graph_def_files",
    name="graph_composed.pbtxt",
    as_text=True
)

#
with tf.Session(graph=g) as sess:
    # Create a FileWriter to write a summary file containing the GraphDef.
    writer = tf.summary.FileWriter(
        graph=g,
        logdir="summary_files/graph_composed",
    )
    
    feed_dict = {
        sg1_a: 4.0, 
        sg1_b: 5.0,
        sg2_d: 5.0
    }
    print(sess.run(sg2_out_calc, feed_dict=feed_dict))
    