# Recursive neural network

## Architectures

### Basic
In the simplest architecture, nodes are combined into parents using a weight matrix (which is shared across the whole network) and a non-linearity such as the  ${\displaystyle \tanh }$  hyperbolic function. If  ${\displaystyle c_{1}}$  and  ${\displaystyle c_{2}}$  are  ${\displaystyle n}$ -dimensional vector representations of nodes, their parent will also be an  ${\displaystyle n}$ -dimensional vector, defined as:

 ${\displaystyle p_{1,2}=\tanh(W[c_{1};c_{2}])}$ 

where  ${\displaystyle W}$  is a learned  ${\displaystyle n\times 2n}$  weight matrix.

This architecture, with a few improvements, has been used for successfully parsing natural scenes, syntactic parsing of natural language sentences, and recursive autoencoding and generative modeling of 3D shape structures in the form of cuboid abstractions.


### Recursive cascade correlation (RecCC)
RecCC is a constructive neural network approach to deal with tree domains  with pioneering applications to chemistry and extension to directed acyclic graphs.


### Unsupervised RNN
A framework for unsupervised RNN has been introduced in 2004.


### Tensor
Recursive neural tensor networks use a single tensor-based composition function for all nodes in the tree.


## Training

### Stochastic gradient descent
Typically, stochastic gradient descent (SGD) is used to train the network. The gradient is computed using backpropagation through structure (BPTS), a variant of backpropagation through time used for recurrent neural networks.


## Properties
The universal approximation capability of RNNs over trees has been proved in literature.


## Related models

### Recurrent neural networks
Recurrent neural networks are recursive artificial neural networks with a certain structure: that of a linear chain. Whereas recursive neural networks operate on any hierarchical structure, combining child representations into parent representations, recurrent neural networks operate on the linear progression of time, combining the previous time step and a hidden representation into the representation for the current time step.


### Tree Echo State Networks
An efficient approach to implement recursive neural networks is given by the Tree Echo State Network within the reservoir computing paradigm.


### Extension to graphs
Extensions to graphs include graph neural network (GNN), Neural Network for Graphs (NN4G), and more recently convolutional neural networks for graphs.