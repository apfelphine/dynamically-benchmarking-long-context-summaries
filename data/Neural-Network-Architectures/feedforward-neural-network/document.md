# Feedforward neural network

## Mathematical foundations

### Activation function
The two historically common activation functions are both sigmoids, and are described by

 ${\displaystyle y(v_{i})=\tanh(v_{i})~~{\text{and}}~~y(v_{i})=(1+e^{-v_{i}})^{-1}.}$ 

The first is a hyperbolic tangent that ranges from -1 to 1, while the other is the logistic function, which is similar in shape but ranges from 0 to 1. Here  ${\displaystyle y_{i}}$  is the output of the  ${\displaystyle i}$ -th node (neuron) and  ${\displaystyle v_{i}}$  is the weighted sum of the input connections. Alternative activation functions have been proposed, including the rectifier and softplus functions. More specialized activation functions include radial basis functions (used in radial basis networks, another class of supervised neural network models).

In recent developments of deep learning, the rectified linear unit (ReLU) is more frequently used as one of the possible ways to overcome the numerical problems related to the sigmoids.


### Learning
Learning occurs by changing connection weights after each piece of data is processed, based on the amount of error in the output compared to the expected result. This is an example of supervised learning, and is carried out through backpropagation.

We can represent the degree of error in an output node  ${\displaystyle j}$  in the  ${\displaystyle n}$ -th data point (training example) by  ${\displaystyle e_{j}(n)=d_{j}(n)-y_{j}(n)}$ , where  ${\displaystyle d_{j}(n)}$  is the desired target value for  ${\displaystyle n}$ -th data point at node  ${\displaystyle j}$ , and  ${\displaystyle y_{j}(n)}$  is the value produced at node  ${\displaystyle j}$  when the  ${\displaystyle n}$ -th data point is given as an input.

The node weights can then be adjusted based on corrections that minimize the error in the entire output for the  ${\displaystyle n}$ -th data point, given by

 ${\displaystyle {\mathcal {E}}(n)={\frac {1}{2}}\sum _{{\text{output node }}j}e_{j}^{2}(n).}$ 

Using gradient descent, the change in each weight  ${\displaystyle w_{ij}}$  is

 ${\displaystyle \Delta w_{ji}(n)=-\eta {\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}y_{i}(n)}$ 

where  ${\displaystyle y_{i}(n)}$  is the output of the previous neuron  ${\displaystyle i}$ , and  ${\displaystyle \eta }$  is the learning rate, which is selected to ensure that the weights quickly converge to a response, without oscillations. In the previous expression,  ${\displaystyle {\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}}$  denotes the partial derivative of the error  ${\displaystyle {\mathcal {E}}(n)}$  according to the weighted sum  ${\displaystyle v_{j}(n)}$  of the input connections of neuron  ${\displaystyle i}$ .

The derivative to be calculated depends on the induced local field  ${\displaystyle v_{j}}$ , which itself varies. It is easy to prove that for an output node this derivative can be simplified to

 ${\displaystyle -{\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}=e_{j}(n)\phi ^{\prime }(v_{j}(n))}$ 

where  ${\displaystyle \phi ^{\prime }}$  is the derivative of the activation function described above, which itself does not vary. The analysis is more difficult for the change in weights to a hidden node, but it can be shown that the relevant derivative is

 ${\displaystyle -{\frac {\partial {\mathcal {E}}(n)}{\partial v_{j}(n)}}=\phi ^{\prime }(v_{j}(n))\sum _{k}-{\frac {\partial {\mathcal {E}}(n)}{\partial v_{k}(n)}}w_{kj}(n).}$ 

This depends on the change in weights of the  ${\displaystyle k}$ th nodes, which represent the output layer. So to change the hidden layer weights, the output layer weights change according to the derivative of the activation function, and so this algorithm represents a backpropagation of the activation function.


## History

### Timeline
- Circa 1800, Legendre (1805) and Gauss (1795) created the simplest feedforward network which consists of a single weight layer with linear activation functions. It was trained by the least squares method for minimising mean squared error, also known as linear regression. Legendre and Gauss used it for the prediction of planetary movement from training data.[11][12][13][14][15]
- In 1943, Warren McCulloch and Walter Pitts proposed the binary artificial neuron as a logical model of biological neural networks.[16]
- In 1958, Frank Rosenblatt proposed the multilayered perceptron model, consisting of an input layer, a hidden layer with randomized weights that did not learn, and an output layer with learnable connections.[17] R. D. Joseph (1960)[18] mentions an even earlier perceptron-like device:[13] "Farley and Clark of MIT Lincoln Laboratory actually preceded Rosenblatt in the development of a perceptron-like device." However, "they dropped the subject."
- In 1960, Joseph[18] also discussed multilayer perceptrons with an adaptive hidden layer. Rosenblatt (1962)[19]: section 16  cited and adopted these ideas, also crediting work by H. D. Block and B. W. Knight. Unfortunately, these early efforts did not lead to a working learning algorithm for hidden units, i.e., deep learning.
- In 1965, Alexey Grigorevich Ivakhnenko and Valentin Lapa published Group Method of Data Handling, the first working deep learning algorithm, a method to train arbitrarily deep neural networks.[20][21] It is based on layer by layer training through regression analysis. Superfluous hidden units are pruned using a separate validation set. Since the activation functions of the nodes are Kolmogorov-Gabor polynomials, these were also the first deep networks with multiplicative units or "gates."[13] It was used to train an eight-layer neural net in 1971.
- In 1967, Shun'ichi Amari reported [22] the first multilayered neural network trained by stochastic gradient descent, which was able to classify non-linearily separable pattern classes. Amari's student Saito conducted the computer experiments, using a five-layered feedforward network with two learning layers.[13]
- In 1970, Seppo Linnainmaa published the modern form of backpropagation in his master thesis (1970).[23][24][13] G.M. Ostrovski et al. republished it in 1971.[25][26] Paul Werbos applied backpropagation to neural networks in 1982[7][27] (his 1974 PhD thesis, reprinted in a 1994 book,[28] did not yet describe the algorithm[26]). In 1986, David E. Rumelhart et al. popularised backpropagation but did not cite the original work.[29][8]
- In 2003, interest in backpropagation networks returned due to the successes of deep learning being applied to language modelling by Yoshua Bengio with co-authors.[30]



### Perceptron
If using a threshold, i.e. a linear activation function,  the resulting linear threshold unit is called a perceptron. (Often the term is used to denote just one of these units.) Multiple parallel non-linear units are able to approximate any continuous function from a compact interval of the real numbers into the interval [−1,1] despite the limited computational power of single unit with a linear threshold function.

Perceptrons can be trained by a simple learning algorithm that is usually called the delta rule. It calculates the errors between calculated output and sample output data, and uses this to create an adjustment to the weights, thus implementing a form of gradient descent.


### Multilayer perceptron
A multilayer perceptron (MLP) is a misnomer for a modern feedforward artificial neural network, consisting of fully connected neurons (hence the synonym sometimes used of fully connected network (FCN)), often with a nonlinear kind of activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable.


## Other feedforward networks
Examples of other feedforward networks include convolutional neural networks and radial basis function networks, which use a different activation function.