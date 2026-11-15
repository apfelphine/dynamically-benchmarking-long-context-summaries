# Multilayer perceptron

## Timeline
- In 1943, Warren McCulloch and Walter Pitts proposed the binary artificial neuron as a logical model of biological neural networks.[11]
- In 1958, Frank Rosenblatt proposed the multilayered perceptron model, consisting of an input layer, a hidden layer with randomized weights that did not learn, and an output layer with learnable connections.[12]
- In 1962, Rosenblatt published many variants and experiments on perceptrons in his book Principles of Neurodynamics, including up to 2 trainable layers by "back-propagating errors".[13] However, it was not the backpropagation algorithm, and he did not have a general method for training multiple layers.


- In 1965, Alexey Grigorevich Ivakhnenko and Valentin Lapa published Group Method of Data Handling. It was one of the first deep learning methods, used to train an eight-layer neural net in 1971.[14][15][16]


- In 1967, Shun'ichi Amari reported [17] the first multilayered neural network trained by stochastic gradient descent, was able to classify non-linearily separable pattern classes. Amari's student Saito conducted the computer experiments, using a five-layered feedforward network with two learning layers.[16]


- Backpropagation was independently developed multiple times in early 1970s. The earliest published instance was Seppo Linnainmaa's master thesis (1970).[18][19][16] Paul Werbos developed it independently in 1971,[20] but had difficulty publishing it until 1982.[21]


- In 1986, David E. Rumelhart et al. popularized backpropagation.[22][23]


- In 2003, interest in backpropagation networks returned due to the successes of deep learning being applied to language modelling by Yoshua Bengio with co-authors.[24]


- In 2021, a very simple NN architecture combining two deep MLPs with skip connections and layer normalizations was designed and called MLP-Mixer; its realizations featuring 19 to 431 millions of parameters were shown to be comparable to vision transformers of similar size on ImageNet and similar image classification tasks.[25]



## Mathematical foundations

### Activation function
If a multilayer perceptron has a linear activation function in all neurons, that is, a linear function that maps the weighted inputs to the output of each neuron, then linear algebra shows that any number of layers can be reduced to a two-layer input-output model. In MLPs some neurons use a nonlinear activation function that was developed to model the frequency of action potentials, or firing, of biological neurons.

The two historically common activation functions are both sigmoids, and are described by

 ${\displaystyle y(v_{i})=\tanh(v_{i})~~{\textrm {and}}~~y(v_{i})=(1+e^{-v_{i}})^{-1}}$ .

The first is a hyperbolic tangent that ranges from −1 to 1, while the other is the logistic function, which is similar in shape but ranges from 0 to 1. Here  ${\displaystyle y_{i}}$  is the output of the  ${\displaystyle i}$ th node (neuron) and  ${\displaystyle v_{i}}$  is the weighted sum of the input connections. Alternative activation functions have been proposed, including the rectifier and softplus functions. More specialized activation functions include radial basis functions (used in radial basis networks, another class of supervised neural network models).

In recent developments of deep learning the rectified linear unit (ReLU) is more frequently used as one of the possible ways to overcome the numerical problems related to the sigmoids.


### Layers
The MLP consists of three or more layers (an input and an output layer with one or more hidden layers) of nonlinearly-activating nodes. Since MLPs are fully connected, each node in one layer connects with a certain weight  ${\displaystyle w_{ij}}$  to every node in the following layer.


### Learning
Learning occurs in the perceptron by changing connection weights after each piece of data is processed, based on the amount of error in the output compared to the expected result. This is an example of supervised learning, and is carried out through backpropagation, a generalization of the least mean squares algorithm in the linear perceptron.

We can represent the degree of error in an output node  ${\displaystyle j}$  in the  ${\displaystyle n}$ th data point (training example) by  ${\displaystyle e_{j}(n)=d_{j}(n)-y_{j}(n)}$ , where  ${\displaystyle d_{j}(n)}$  is the desired target value for  ${\displaystyle n}$ th data point at node  ${\displaystyle j}$ , and  ${\displaystyle y_{j}(n)}$  is the value produced by the perceptron at node  ${\displaystyle j}$  when the  ${\displaystyle n}$ th data point is given as an input.

The node weights can then be adjusted based on corrections that minimize the error in the entire output for the  ${\displaystyle n}$ th data point, given by

 ${\displaystyle {\mathcal {E}}(n)={\frac {1}{2}}\sum _{{\text{output node }}j}e_{j}^{2}(n)}$ .

Using gradient descent, the change in each weight  ${\displaystyle w_{ij}}$  is

 ${\displaystyle \Delta w_{ji}(n)=-\eta {\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}y_{i}(n)}$ 

where  ${\displaystyle y_{i}(n)}$  is the output of the previous neuron  ${\displaystyle i}$ , and  ${\displaystyle \eta }$  is the learning rate, which is selected to ensure that the weights quickly converge to a response, without oscillations. In the previous expression,  ${\displaystyle {\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}}$  denotes the partial derivate of the error  ${\displaystyle {\mathcal {E}}(n)}$  according to the weighted sum  ${\displaystyle v_{j}(n)}$  of the input connections of neuron  ${\displaystyle i}$ .

The derivative to be calculated depends on the induced local field  ${\displaystyle v_{j}}$ , which itself varies. It is easy to prove that for an output node this derivative can be simplified to

 ${\displaystyle -{\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}=e_{j}(n)\phi ^{\prime }(v_{j}(n))}$ 

where  ${\displaystyle \phi ^{\prime }}$  is the derivative of the activation function described above, which itself does not vary. The analysis is more difficult for the change in weights to a hidden node, but it can be shown that the relevant derivative is

 ${\displaystyle -{\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}=\phi ^{\prime }(v_{j}(n))\sum _{k}-{\frac {\partial {\mathcal {E}}(n)}{\partial v_{k}(n)}}w_{kj}(n)}$ .

This depends on the change in weights of the  ${\displaystyle k}$ th nodes, which represent the output layer. So to change the hidden layer weights, the output layer weights change according to the derivative of the activation function, and so this algorithm represents a backpropagation of the activation function.