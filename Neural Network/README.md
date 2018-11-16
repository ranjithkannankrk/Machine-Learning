# Neural Network

This is an implementation of neural network for the MNIST handwritten digits recognition problem.

The program takes two inputs
1. The zipped MNIST image file consisting of samples for 10000 digit representations
2. The zipped MNIST lable file for the respective digit representation in the image file.

Since we have a training data int_train of 48000 samples, where each sample has features arranged in 28X28 matrix. To fit the problem for neural net designing, the each sample feature representation was converted from a 28X28 representation to 784 feature representation which was done using numpy’s reshape feature. Since we have 784 features the neural network was configured to have 784 nodes in the first layer, which is the input layer. Here a weight is initialized for each feature and then is fed along with the input nodes in the first layer.

The second layer which is the hidden layer consists of 50 nodes which have feedforwarding and backpropagation implementations within that, inclusive of the gradient computation. The feedforwarding is used to pass the input and compute new activations. Here in the design, sigmoid, tanh and ReLU activation function are evaluated. Backpropagation is used to update each of the weights in the network so that they cause the actual output to be closer to the target output, thereby minimizing the error for each output neuron and the network as a whole. It is a method to stop from overfitting our model, so the model is more generalized. All 784 node’s outputs are fed into the 50 nodes in the hidden layer. These nodes take in the weights from the previous layer and after computation come up with new weights which are given as outputs to the output layer.

The output layer in the design consists of 10 nodes, one node for each class, so in our case in MNIST there are 10 classes and hence the number of nodes in the output layer is 10.

The best configuration parameters are listed as follows
* Activation function --> ReLU
* Number of Hidden layer nodes --> 50
* Loss function --> log loss
* Input feature representation --> Normalized
