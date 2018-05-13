mkdir frozen_graphs

# Freezes the GraphDef proto - converts all TF Variable tensors into TF Constant tensors.
python freeze_graph.py \
  --input_graph exported_graphs/graphdef.pbtxt \
  --input_checkpoint saved_model/linear_regression.ckpt-1900 \
  --output_node_names linear_regression/add \
  --output_graph frozen_graphs/frozen_linear_regression.pb