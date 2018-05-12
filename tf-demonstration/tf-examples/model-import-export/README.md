# model-import-export

## model_producer

The `model_producer.py` script runs the model training phase and persists the 
estimated model with the estimated values of the model parameters.

Run the `model_producer.py` by the following command:

```python
python model_producer.py
```

The following is the output from running the script:

```
epoch: 0, loss: 63893.72099813372, slope: 1.8000000119204493, intercept: 1.8000000095987017
epoch: 100, loss: 133.52213423614387, slope: 1.46548287548405, intercept: -0.7200838280311855
epoch: 200, loss: 36.25790133238273, slope: 1.4820152036979868, intercept: -1.8113076304160947
epoch: 300, loss: 9.915691890557488, slope: 1.4905977329850604, intercept: -2.377848457259814
epoch: 400, loss: 2.7181037488265005, slope: 1.4950809850290234, intercept: -2.6737987325979167
epoch: 500, loss: 0.7467795053285284, slope: 1.4974268313816979, intercept: -2.8286546454616235
epoch: 600, loss: 0.20650796732969873, slope: 1.498654849864709, intercept: -2.90971973263873
epoch: 700, loss: 0.05841187523842594, slope: 1.4992977813855068, intercept: -2.9521615396844005
epoch: 800, loss: 0.01781465079992764, slope: 1.499634401033928, intercept: -2.974382798848525
epoch: 900, loss: 0.006685682453091426, slope: 1.4998106466397836, intercept: -2.98601729325952
epoch: 1000, loss: 0.003634873113008704, slope: 1.4999029246573117, intercept: -2.99210883854396
epoch: 1100, loss: 0.0027985469852491313, slope: 1.4999512392720618, intercept: -2.995298229390137
epoch: 1200, loss: 0.0025692827123589253, slope: 1.4999765356808283, intercept: -2.996968120271713
epoch: 1300, loss: 0.002506433891717342, slope: 1.499989780292916, intercept: -2.9978424363520593
epoch: 1400, loss: 0.002489204978165108, slope: 1.4999967148642828, intercept: -2.998300207951286
epoch: 1500, loss: 0.0024844819704657505, slope: 1.5000003456448534, intercept: -2.9985398865327526
epoch: 1600, loss: 0.0024831872398546322, slope: 1.500002246637268, intercept: -2.9986653766736264
epoch: 1700, loss: 0.0024828323119225407, slope: 1.5000032419527631, intercept: -2.9987310803980276
epoch: 1800, loss: 0.002482735014588079, slope: 1.5000037630768492, intercept: -2.9987654813427604
epoch: 1900, loss: 0.0024827083422117174, slope: 1.5000040359253222, intercept: -2.9987834928787764
```

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