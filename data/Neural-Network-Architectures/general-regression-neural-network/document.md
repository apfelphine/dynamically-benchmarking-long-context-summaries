# General regression neural network

## Mathematical representation
${\displaystyle Y(x)={\frac {\sum _{k=1}^{N}y_{k}K(x,x_{k})}{\sum _{k=1}^{N}K(x,x_{k})}}}$ 

where:

- ${\displaystyle Y(x)}$  is the prediction value of input  ${\displaystyle x}$
- ${\displaystyle y_{k}}$  is the activation weight for the pattern layer neuron at  ${\displaystyle k}$
- ${\displaystyle K(x,x_{k})}$  is the Radial basis function kernel (Gaussian kernel) as formulated below.



### Gaussian Kernel
${\displaystyle K(x,x_{k})=e^{-d_{k}/2\sigma ^{2}},\qquad d_{k}=(x-x_{k})^{T}(x-x_{k})}$ 

where  ${\displaystyle d_{k}}$  is the squared euclidean distance between the training samples  ${\displaystyle x_{k}}$  and the input  ${\displaystyle x.}$ 


## Implementation
GRNN has been implemented in many computer languages including MATLAB, R- programming language, Python (programming language) and Node.js.

Neural networks (specifically Multi-layer Perceptron) can delineate non-linear patterns in data by combining with generalized linear models by considering distribution of outcomes (sightly different from original GRNN). There have been several successful developments, including Poisson regression, ordinal logistic regression, quantile regression and multinomial logistic regression that described by Fallah in 2009.



## Advantages and disadvantages
Similar to RBFNN, GRNN has the following advantages:

- Single-pass learning so no backpropagation is required.
- High accuracy in the estimation since it uses Gaussian functions.
- It can handle noises in the inputs.
- It requires relatively few data to train.


The main disadvantages of GRNN are:

- Its size can be huge, which would make it computationally expensive.
- There is no optimal method to improve it.