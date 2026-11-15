# Restricted Boltzmann machine

## Structure
The standard type of RBM has binary-valued (Boolean) hidden and visible units, and consists of a matrix of weights  ${\displaystyle W}$  of size  ${\displaystyle m\times n}$ . Each weight element  ${\displaystyle (w_{i,j})}$  of the matrix is associated with the connection between the visible (input) unit  ${\displaystyle v_{i}}$  and the hidden unit  ${\displaystyle h_{j}}$ . In addition, there are bias weights (offsets)  ${\displaystyle a_{i}}$  for  ${\displaystyle v_{i}}$  and  ${\displaystyle b_{j}}$  for  ${\displaystyle h_{j}}$ . Given the weights and biases, the energy of a configuration (pair of Boolean vectors) (v,h) is defined as

 ${\displaystyle E(v,h)=-\sum _{i}a_{i}v_{i}-\sum _{j}b_{j}h_{j}-\sum _{i}\sum _{j}v_{i}w_{i,j}h_{j}}$ 

or, in matrix notation,

 ${\displaystyle E(v,h)=-a^{\mathrm {T} }v-b^{\mathrm {T} }h-v^{\mathrm {T} }Wh.}$ 

This energy function is analogous to that of a Hopfield network. As with general Boltzmann machines, the joint probability distribution for the visible and hidden vectors is defined in terms of the energy function as follows,

 ${\displaystyle P(v,h)={\frac {1}{Z}}e^{-E(v,h)}}$ 

where  ${\displaystyle Z}$  is a partition function defined as the sum of  ${\displaystyle e^{-E(v,h)}}$  over all possible configurations, which can be interpreted as a normalizing constant to ensure that the probabilities sum to 1. The marginal probability of a visible vector is the sum of  ${\displaystyle P(v,h)}$  over all possible hidden layer configurations,

 ${\displaystyle P(v)={\frac {1}{Z}}\sum _{\{h\}}e^{-E(v,h)}}$ ,

and vice versa. Since the underlying graph structure of the RBM is bipartite (meaning there are no intra-layer connections), the hidden unit activations are mutually independent given the visible unit activations. Conversely, the visible unit activations are mutually independent given the hidden unit activations. That is, for m visible units and n hidden units, the conditional probability of a configuration of the visible units v, given a configuration of the hidden units h, is

 ${\displaystyle P(v|h)=\prod _{i=1}^{m}P(v_{i}|h)}$ .

Conversely, the conditional probability of h given v is

 ${\displaystyle P(h|v)=\prod _{j=1}^{n}P(h_{j}|v)}$ .

The individual activation probabilities are given by

 ${\displaystyle P(h_{j}=1|v)=\sigma \left(b_{j}+\sum _{i=1}^{m}w_{i,j}v_{i}\right)}$  and  ${\displaystyle \,P(v_{i}=1|h)=\sigma \left(a_{i}+\sum _{j=1}^{n}w_{i,j}h_{j}\right)}$ 

where  ${\displaystyle \sigma }$  denotes the logistic sigmoid.

The visible units of Restricted Boltzmann Machine can be multinomial, although the hidden units are Bernoulli. In this case, the logistic function for visible units is replaced by the softmax function

 ${\displaystyle P(v_{i}^{k}=1|h)={\frac {\exp(a_{i}^{k}+\Sigma _{j}W_{ij}^{k}h_{j})}{\Sigma _{k'=1}^{K}\exp(a_{i}^{k'}+\Sigma _{j}W_{ij}^{k'}h_{j})}}}$ 

where K is the number of discrete values that the visible values have. They are applied in topic modeling, and recommender systems.


### Relation to other models
Restricted Boltzmann machines are a special case of Boltzmann machines and Markov random fields.

The graphical model of RBMs corresponds to that of factor analysis.


## Training algorithm
Restricted Boltzmann machines are trained to maximize the product of probabilities assigned to some training set  ${\displaystyle V}$  (a matrix, each row of which is treated as a visible vector  ${\displaystyle v}$ ),

 ${\displaystyle \arg \max _{W}\prod _{v\in V}P(v)}$ 

or equivalently, to maximize the expected log probability of a training sample  ${\displaystyle v}$  selected randomly from  ${\displaystyle V}$ :

 ${\displaystyle \arg \max _{W}\mathbb {E} \left[\log P(v)\right]}$ 

The algorithm most often used to train RBMs, that is, to optimize the weight matrix  ${\displaystyle W}$ , is the contrastive divergence (CD) algorithm due to Hinton, originally developed to train PoE (product of experts) models.
The algorithm performs Gibbs sampling and is used inside a gradient descent procedure (similar to the way backpropagation is used inside such a procedure when training feedforward neural nets) to compute weight update.

The basic, single-step contrastive divergence (CD-1) procedure for a single sample can be summarized as follows:

1. Take a training sample v, compute the probabilities of the hidden units and sample a hidden activation vector h from this probability distribution.
2. Compute the outer product of v and h and call this the positive gradient.
3. From h, sample a reconstruction v' of the visible units, then resample the hidden activations h' from this. (Gibbs sampling step)
4. Compute the outer product of v' and h' and call this the negative gradient.
5. Let the update to the weight matrix  ${\displaystyle W}$  be the positive gradient minus the negative gradient, times some learning rate:  ${\displaystyle \Delta W=\epsilon (vh^{\mathsf {T}}-v'h'^{\mathsf {T}})}$ .
6. Update the biases a and b analogously:  ${\displaystyle \Delta a=\epsilon (v-v')}$ ,  ${\displaystyle \Delta b=\epsilon (h-h')}$ .


A Practical Guide to Training RBMs written by Hinton can be found on his homepage.


## Stacked Restricted Boltzmann Machine
- The difference between the Stacked Restricted Boltzmann Machines and RBM is that RBM has lateral connections within a layer that are prohibited to make analysis tractable. On the other hand, the Stacked Boltzmann consists of a combination of an unsupervised three-layer network with symmetric weights and a supervised fine-tuned top layer for recognizing three classes.
- The usage of Stacked Boltzmann is to understand Natural languages, retrieve documents, image generation, and classification. These functions are trained with unsupervised pre-training and/or supervised fine-tuning. Unlike the undirected symmetric top layer, with a two-way unsymmetric layer for connection for RBM. The restricted Boltzmann's connection is three-layers with asymmetric weights, and two networks are combined into one.
- Stacked Boltzmann does share similarities with RBM, the neuron for Stacked Boltzmann is a stochastic binary Hopfield neuron, which is the same as the Restricted Boltzmann Machine. The energy from both Restricted Boltzmann and RBM is given by Gibb's probability measure:  ${\displaystyle E=-{\frac {1}{2}}\sum _{i,j}{w_{ij}{s_{i}}{s_{j}}}+\sum _{i}{\theta _{i}}{s_{i}}}$ . The training process of Restricted Boltzmann is similar to RBM. Restricted Boltzmann train one layer at a time and approximate equilibrium state with a 3-segment pass, not performing back propagation. Restricted Boltzmann uses both supervised and unsupervised on different RBM for pre-training for classification and recognition. The training uses contrastive divergence with Gibbs sampling: Δwij = e*(pij - p'ij)
- The restricted Boltzmann's strength is it performs a non-linear transformation so it's easy to expand, and can give a hierarchical layer of features. The Weakness is that it has complicated calculations of integer and real-valued neurons. It does not follow the gradient of any function, so the approximation of Contrastive divergence to maximum likelihood is improvised.[14]



## Literature
- Fischer, Asja; Igel, Christian (2012), "An Introduction to Restricted Boltzmann Machines", Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications, Lecture Notes in Computer Science, vol. 7441, Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 14–36, doi:10.1007/978-3-642-33275-3_2, ISBN 978-3-642-33274-6