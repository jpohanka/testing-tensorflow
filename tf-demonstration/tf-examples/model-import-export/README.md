# model-import-export


## MetaGraphDef proto

The `MetaGraphDef` proto is composed of:

  * `MetaInfoDef` proto - contains the graph metadata (e.g. model version)
  * `GraphDef` proto - represents the graph of operations
  * `SaverDef` proto - the configuration of a Saver
  * `CollectionDef` map - describes additional components of the model, such as Variables, tf.train.QueueRunner, etc.
  * `SignatureDef` map - contains the signature information of each Tensor in the graph (node inputs/outputs, type, shape, sparsity etc.)
  * `AssetFileDef` proto - 

## The Graph class

The Python `Graph` class is created in the `model_producer.py` script by the following line:

```python
g = tf.Graph()
```

The `Graph` class is a wrapper class to enable easy maniputaion with the TF graph in Python.

### as_graph_def()

```python
g.as_graph_def()
```

The function returns a `GraphDef` proto:

```
node {
  name: "inputs/Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_DOUBLE
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: -1
        }
      }
    }
  }
}
...
```

Check out the `graphdef.pbtxt` file, where the `GraphDef` proto of the current model is stored.

### get_all_collection_keys()

```python
g.get_all_collection_keys()
```

The output will be the following:

```
['train_op', 'summaries', 'variables', 'trainable_variables']
```

### get_operations()

```python
g.get_operations()
```

The output will be all operations in the graph:

```
[<tf.Operation 'inputs/Placeholder' type=Placeholder>,
 <tf.Operation 'inputs/Placeholder_1' type=Placeholder>,
 <tf.Operation 'model_variables/Const' type=Const>,
 <tf.Operation 'model_variables/slope' type=VariableV2>,
 <tf.Operation 'model_variables/slope/Assign' type=Assign>,
 <tf.Operation 'model_variables/slope/read' type=Identity>,
 ...
```

Check out the `get_operations.txt` file, where the full list of operations is stored.