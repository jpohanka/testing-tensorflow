import tensorflow as tf

g = tf.Graph()

# Define sub-graph 1:
with g.as_default():
    a = tf.placeholder(shape=[], dtype=tf.float64, name="a")
    b = tf.placeholder(shape=[], dtype=tf.float64, name="b")
    _output = a + b
    output = tf.identity(_output, name="out_sum")

# Save the GraphDef proto message and generate a simple summary file for the TensorBoard.
with tf.Session(graph=g) as sess:
    # Save the GraphDef proto twice - in binary and in human-readable pbtxt format.
    tf.train.write_graph(
        graph_or_graph_def=g,
        logdir="graph_def_files",
        name="subgraph1_graph_def.pb",
        as_text=False
    )
    tf.train.write_graph(
        graph_or_graph_def=g,
        logdir="graph_def_files",
        name="subgraph1_graph_def.pbtxt",
        as_text=True
    )
    
    # Create a FileWriter to write a summary file containing the GraphDef.
    writer = tf.summary.FileWriter(
        graph=g,
        logdir="summary_files/subgraph1",
    )
    
    # Close the FileWriter and write the data to disk.
    writer.close()