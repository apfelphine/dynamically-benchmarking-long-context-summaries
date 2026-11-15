# Siamese neural network

## Learning
Learning in twin networks can be done with triplet loss or contrastive loss. For learning by triplet loss a baseline vector (anchor image) is compared against a positive vector (truthy image) and a negative vector (falsy image). The negative vector will force learning in the network, while the positive vector will act like a regularizer. For learning by contrastive loss there must be a weight decay to regularize the weights, or some similar operation like a normalization.

A distance metric for a loss function may have the following properties

- Non-negativity:  ${\displaystyle \delta (x,y)\geq 0}$
- Identity of Non-discernibles:  ${\displaystyle \delta (x,y)=0\iff x=y}$
- Commutativity:  ${\displaystyle \delta (x,y)=\delta (y,x)}$
- Triangle inequality:  ${\displaystyle \delta (x,z)\leq \delta (x,y)+\delta (y,z)}$


In particular, the triplet loss algorithm is often defined with squared Euclidean (which unlike Euclidean, does not have triangle inequality) distance at its core.


### Predefined metrics, Euclidean distance metric
The common learning goal is to minimize a distance metric for similar objects and maximize for distinct ones. This gives a loss function like

 ${\displaystyle {\begin{aligned}\delta (x^{(i)},x^{(j)})={\begin{cases}\min \ \|\operatorname {f} \left(x^{(i)}\right)-\operatorname {f} \left(x^{(j)}\right)\|\,,i=j\\\max \ \|\operatorname {f} \left(x^{(i)}\right)-\operatorname {f} \left(x^{(j)}\right)\|\,,i\neq j\end{cases}}\end{aligned}}}$ 
 ${\displaystyle i,j}$  are indexes into a set of vectors
 ${\displaystyle \operatorname {f} (\cdot )}$  function implemented by the twin network

The most common distance metric used is Euclidean distance, in case of which the loss function can be rewritten in matrix form as

 ${\displaystyle \operatorname {\delta } (\mathbf {x} ^{(i)},\mathbf {x} ^{(j)})\approx (\mathbf {x} ^{(i)}-\mathbf {x} ^{(j)})^{T}(\mathbf {x} ^{(i)}-\mathbf {x} ^{(j)})}$ 


### Learned metrics, nonlinear distance metric
A more general case is where the output vector from the twin network is passed through additional network layers implementing non-linear distance metrics.

 ${\displaystyle {\begin{aligned}{\text{if}}\,i=j\,{\text{then}}&\,\operatorname {\delta } \left[\operatorname {f} \left(x^{(i)}\right),\,\operatorname {f} \left(x^{(j)}\right)\right]\,{\text{is small}}\\{\text{otherwise}}&\,\operatorname {\delta } \left[\operatorname {f} \left(x^{(i)}\right),\,\operatorname {f} \left(x^{(j)}\right)\right]\,{\text{is large}}\end{aligned}}}$ 
 ${\displaystyle i,j}$  are indexes into a set of vectors
 ${\displaystyle \operatorname {f} (\cdot )}$ function implemented by the twin network
 ${\displaystyle \operatorname {\delta } (\cdot )}$ function implemented by the network joining outputs from the twin network

On a matrix form the previous is often approximated as a Mahalanobis distance for a linear space as

 ${\displaystyle \operatorname {\delta } (\mathbf {x} ^{(i)},\mathbf {x} ^{(j)})\approx (\mathbf {x} ^{(i)}-\mathbf {x} ^{(j)})^{T}\mathbf {M} (\mathbf {x} ^{(i)}-\mathbf {x} ^{(j)})}$ 

This can be further subdivided in at least Unsupervised learning and Supervised learning.


### Learned metrics, half-twin networks
This form also allows the twin network to be more of a half-twin, implementing a slightly different functions

 ${\displaystyle {\begin{aligned}{\text{if}}\,i=j\,{\text{then}}&\,\operatorname {\delta } \left[\operatorname {f} \left(x^{(i)}\right),\,\operatorname {g} \left(x^{(j)}\right)\right]\,{\text{is small}}\\{\text{otherwise}}&\,\operatorname {\delta } \left[\operatorname {f} \left(x^{(i)}\right),\,\operatorname {g} \left(x^{(j)}\right)\right]\,{\text{is large}}\end{aligned}}}$ 
 ${\displaystyle i,j}$  are indexes into a set of vectors
 ${\displaystyle \operatorname {f} (\cdot ),\operatorname {g} (\cdot )}$ function implemented by the half-twin network
 ${\displaystyle \operatorname {\delta } (\cdot )}$ function implemented by the network joining outputs from the twin network


## Twin networks for object tracking
Twin networks have been used in object tracking because of its unique two tandem inputs and similarity measurement. In object tracking, one input of the twin network is user pre-selected exemplar image, the other input is a larger search image. The twin network's job is to locate the exemplar inside of the search image. By measuring the similarity between exemplar and each part of the search image, a map of similarity score can be given by the twin network. Furthermore, using a Fully Convolutional Network, the process of computing each sector's similarity score can be replaced with only one cross correlation layer.

After being first introduced in 2016, Twin fully convolutional network has been used in many High-performance Real-time Object Tracking Neural Networks.  Like CFnet, StructSiam, SiamFC-tri, DSiam, SA-Siam, SiamRPN, DaSiamRPN, Cascaded SiamRPN, SiamMask, SiamRPN++, Deeper and Wider SiamRPN.