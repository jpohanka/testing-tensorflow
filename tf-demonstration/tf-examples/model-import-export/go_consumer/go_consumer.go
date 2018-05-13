package main

import (
	"fmt"
	"io/ioutil"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	path         = "frozen_graphs"
	graphDefFile = "frozen_linear_regression.pb"
)

// The following example does a simple loading of the persisted GraphDef proto,
// initializes a TF session a runs a simple calculation task.
func main() {
	var err error

	// Read the serialized GraphDef proto from the file.
	model, err := ioutil.ReadFile(path + "/" + graphDefFile)
	if err != nil {
		log.Println(err)
	}

	// Create a Graph from the imported GraphDef proto.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Println(err)
	}

	// Create a new TF session.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// Create a input Tensor for the 'x_sample' placeholder.
	x_sample, err := tf.NewTensor([]float64{0.0, 1.0, -1.0, 2})
	if err != nil {
		log.Fatal(err)
	}

	// Define a feed_dict mapping.
	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[graph.Operation("inputs/x_sample").Output(0)] = x_sample

	// Define the list of nodes which values we are going to calculate.
	fetches := []tf.Output{
		graph.Operation("linear_regression/add").Output(0),
	}

	// Run the model calculation with the x_sample Tensor define above.
	output, err := session.Run(feeds, fetches, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Print out the calculation results.
	fmt.Println("The calculation results :", output[0].Value())
}
