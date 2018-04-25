### Status
[![Build Status](https://travis-ci.org/jpohanka/testing-tensorflow.svg?branch=master)](https://travis-ci.org/jpohanka/testing-tensorflow)

# testing-tensorflow

TensorFlow, in the most general terms, is a software framework for numerical computations based on **dataflow graphs**. It is designed primarily, however, as an interface
for expressing and implementing machine learning algorithms, chief among them
deep neural networks.

The core of TensorFlow is in C++, and it has two primary high-level **frontend languages** and interfaces for expressing and executing the computation graphs. The most
developed frontend is in **Python**, used by most researchers and data scientists. The
C++ frontend provides quite a **low-level API**, useful for efficient execution in embedded systems and other scenarios.

Aside from its portability, another key aspect of TensorFlow is its flexibility, allowing
researchers and data scientists to express models with relative ease. It is sometimes
revealing to think of modern deep learning research and practice as playing with
“LEGO-like” bricks, replacing blocks of the network with others and seeing what hap‐
pens, and at times designing new blocks.

TensorFlow provides helpful tools to use these modular blocks, combined with a flexible
API that enables the writing of new ones. In deep learning, networks are trained with
a feedback process called backpropagation based on gradient descent optimization.
TensorFlow flexibly supports many optimization algorithms, all with automatic dif‐
ferentiation—the user does not need to specify any gradients in advance, since Ten‐
sorFlow derives them automatically based on the computation graph and loss
function provided by the user. To monitor, debug, and visualize the training process,
and to streamline experiments, TensorFlow comes with **TensorBoard**.

Key enablers of TensorFlow’s flexibility for data scientists and researchers are high-
level abstraction libraries. In state-of-the-art deep neural nets for computer vision or
NLU, writing TensorFlow code can take a toll—it can become a complex, lengthy, and
cumbersome endeavor. Abstraction libraries such as Keras and TF-Slim offer simpli‐
fied high-level access to the “**LEGO bricks**” in the lower-level library, helping to
streamline the construction of the dataflow graphs, training them, and running infer‐
ence. Another key enabler for data scientists and engineers is the pretrained models
that come with TF-Slim and TensorFlow. These models were trained on massive
amounts of data with great computational resources, which are often hard to come by
and in any case require much effort to acquire and set up. Using Keras or TF-Slim, for
example, with just a few lines of code it is possible to use these advanced models for
inference on incoming data, and also to fine-tune the models to adapt to new data.

TensorFlow allows us to implement machine learning algorithms by creating and
computing operations that interact with one another. These interactions form what
we call a “**computation graph**,” with which we can intuitively represent complicated
functional architectures.

TensorFlow optimizes its computations based on the graph’s connectivity. Each graph
has its own set of node dependencies.

Working with TensorFlow involves two main phases: 

1. Constructing the computation graph.
2. Executing the computation graph.
    * Creating a TensorFlow **Session**.


* **Nodes** are operations.
* **Edges** are Tensor objects.

The basic units of data that pass through a graph are numerical, Boolean, or string
elements.

There are three main building blocks of TensorFlow:

* Variables
* Placeholders
* Optimization

The optimization process serves to tune the parameters of some given model. For
that purpose, TensorFlow uses special objects called **Variables**.

* First we call the `tf.Variable()` function in order to create a Variable and define what value it will be initialized with.
* We then have to explicitly perform an initialization operation by running the session with the `tf.global_variables_initializer()` method, which allocates the memory for the
Variable and sets its initial values.

TensorFlow has designated built-in structures for feeding input values. These structures are called **placeholders**.

TensorFlow also has a built-in class we can use for the same purpose as in the previ‐
ous examples, offering additional useful features as we will see shortly. This class is
referred to as the **Saver** class.

**Saver** adds operations that allow us to save and restore the model’s parameters by
using binary files called checkpoint files, mapping the tensor values to the names of
the variables. Unlike the method used in the previous section, here we don’t have to
keep track of our parameters— Saver does it automatically for us.

Saver also allows us to restore the graph without having to reconstruct it by 
generating `.meta` checkpoint files containing all the required information about it.

The trained model will be serialized and exported to two files:

* one that contains information about our variables
* one that holds information about our graph and other metadata

The information about the graph and how to incorporate the saved weights in it
(metainformation) is referred to as the `MetaGraphDef` . This information is serialized
—transformed to a string—using **protocol buffers**, and it includes several parts. 
The information about the architecture of the network is kept in `graph_def`.

In order to load the saved graph, we use `tf.train.import_meta_graph()` , passing
the name of the checkpoint file we want (with the `.meta` extension). TensorFlow
already knows what to do with the restored weights, since this information is also
kept.

Simply importing the graph and restoring the weights, however, is **not enough** and
will result in an **error**. The reason is that importing the model and restoring the
weights doesn’t give us additional access to the variables used as arguments when
running the session ( `fetches` and keys of `feed_dict` )—the model doesn’t know what
the inputs and outputs are, what measures we wish to calculate, etc.

One way to solve this problem is by saving them in a **collection**. A collection is a TensorFlow object similar to a dictionary, in which we can keep our graph components
in an orderly, accessible fashion.

As shown, the `saver.save()` method automatically saves the graph architecture
together with the weights’ checkpoints. We can also save the graph explicitly using
`saver.export_meta.graph()` , and then add a collection (passed as the second argu‐
ment).

## Tensorflow Serving

TensorFlow Serving, written in C++, is a high-performance serving framework with
which we can deploy our model in a production setting. It makes our model usable
for production by enabling client software to access it and pass inputs through Serving’s API.

In a nutshell, this is how Serving’s architecture works:

* A module called **Source** identifies new models to be loaded by monitoring
plugged-in filesystems, which contain our models and their associated information that we exported upon creation. Source includes submodules that periodically inspect the filesystem and determine the latest relevant model versions.
* When it identifies a new model version, **source** creates a **loader**. The loader passes its **servables** (objects that clients use to perform computations such as predictions) to a **manager**. The manager handles the full life cycle of servables (loading, unloading, and serving) according to a version policy (gradual rollout, reverting versions, etc.).

Serving requires a specific serialization format and metadata, so we cannot simply use the Saver class, as we saw at the beginning of this chapter.

The following are the steps of exporting TensorFlow models:

1. Define the model.
2. Create a model **builder** instance.
3. Have the metadata (model, method, inputs and outputs, etc.) defined in the
builder in a serialized format (this is referred to as `SignatureDef`).
4. Save the model by using the builder.

`SavedModelBuilder` exports serialized files representing the model in the required format.

The the input (shape of the input tensor of the graph) and output
(tensor of the prediction) **signatures** serve a similar purpose as collections.

The variables and meta-graph information is added to the builder instance via the `SavedModelBuilder.add_meta_graph_and_variables()` method.

We need to pass four arguments: 

1. the session
2. tags (to “serve” or “train”)
3. the signature map
4. some initializations

