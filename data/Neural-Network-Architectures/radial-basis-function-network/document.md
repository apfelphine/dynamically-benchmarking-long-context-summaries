# Radial basis function network

## Network architecture
Radial basis function (RBF) networks typically have three layers: an input layer, a hidden layer with a non-linear RBF activation function and a linear output layer. The input can be modeled as a vector of real numbers  ${\displaystyle \mathbf {x} \in \mathbb {R} ^{n}}$ . The output of the network is then a scalar function of the input vector,  ${\displaystyle \varphi :\mathbb {R} ^{n}\to \mathbb {R} }$ , and is given by

 ${\displaystyle \varphi (\mathbf {x} )=\sum _{i=1}^{N}a_{i}\rho (||\mathbf {x} -\mathbf {c} _{i}||)}$ 

where  ${\displaystyle N}$  is the number of neurons in the hidden layer,  ${\displaystyle \mathbf {c} _{i}}$  is the center vector for neuron  ${\displaystyle i}$ , and  ${\displaystyle a_{i}}$  is the weight of neuron  ${\displaystyle i}$  in the linear output neuron. Functions that depend only on the distance from a center vector are radially symmetric about that vector, hence the name radial basis function. In the basic form, all inputs are connected to each hidden neuron. The norm is typically taken to be the Euclidean distance (although the Mahalanobis distance appears to perform better with pattern recognition) and the radial basis function is commonly taken to be Gaussian

 ${\displaystyle \rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}=\exp \left[-\beta _{i}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert ^{2}\right]}$ .

The Gaussian basis functions are local to the center vector in the sense that

 ${\displaystyle \lim _{||x||\to \infty }\rho (\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert )=0}$ 

i.e. changing parameters of one neuron has only a small effect for input values that are far away from the center of that neuron.

Given certain mild conditions on the shape of the activation function, RBF networks are universal approximators on a compact subset of  ${\displaystyle \mathbb {R} ^{n}}$ . This means that an RBF network with enough hidden neurons can approximate any continuous function on a closed, bounded set with arbitrary precision.

The parameters  ${\displaystyle a_{i}}$ ,  ${\displaystyle \mathbf {c} _{i}}$ , and  ${\displaystyle \beta _{i}}$  are determined in a manner that optimizes the fit between  ${\displaystyle \varphi }$  and the data.


### Normalization

#### Normalized architecture
In addition to the above unnormalized architecture, RBF networks can be normalized. In this case the mapping is

 ${\displaystyle \varphi (\mathbf {x} )\ {\stackrel {\mathrm {def} }{=}}\ {\frac {\sum _{i=1}^{N}a_{i}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}}=\sum _{i=1}^{N}a_{i}u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

where

 ${\displaystyle u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}\ {\stackrel {\mathrm {def} }{=}}\ {\frac {\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{j=1}^{N}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{j}\right\Vert {\big )}}}}$ 

is known as a normalized radial basis function.


#### Theoretical motivation for normalization
There is theoretical justification for this architecture in the case of stochastic data flow. Assume a stochastic kernel approximation for the joint probability density

 ${\displaystyle P\left(\mathbf {x} \land y\right)={1 \over N}\sum _{i=1}^{N}\,\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}\,\sigma {\big (}\left\vert y-e_{i}\right\vert {\big )}}$ 

where the weights  ${\displaystyle \mathbf {c} _{i}}$   and  ${\displaystyle e_{i}}$  are exemplars from the data and we require the kernels to be normalized

 ${\displaystyle \int \rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}\,d^{n}\mathbf {x} =1}$ 

and

 ${\displaystyle \int \sigma {\big (}\left\vert y-e_{i}\right\vert {\big )}\,dy=1}$ .

The probability densities in the input and output spaces are

 ${\displaystyle P\left(\mathbf {x} \right)=\int P\left(\mathbf {x} \land y\right)\,dy={1 \over N}\sum _{i=1}^{N}\,\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

and

The expectation of y given an input  ${\displaystyle \mathbf {x} }$  is

 ${\displaystyle \varphi \left(\mathbf {x} \right)\ {\stackrel {\mathrm {def} }{=}}\ E\left(y\mid \mathbf {x} \right)=\int y\,P\left(y\mid \mathbf {x} \right)dy}$ 

where

 ${\displaystyle P\left(y\mid \mathbf {x} \right)}$ 

is the conditional probability of y given  ${\displaystyle \mathbf {x} }$ .
The conditional probability is related to the joint probability through Bayes' theorem

 ${\displaystyle P\left(y\mid \mathbf {x} \right)={\frac {P\left(\mathbf {x} \land y\right)}{P\left(\mathbf {x} \right)}}}$ 

which yields

 ${\displaystyle \varphi \left(\mathbf {x} \right)=\int y\,{\frac {P\left(\mathbf {x} \land y\right)}{P\left(\mathbf {x} \right)}}\,dy}$ .

This becomes

 ${\displaystyle \varphi \left(\mathbf {x} \right)={\frac {\sum _{i=1}^{N}e_{i}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}}=\sum _{i=1}^{N}e_{i}u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

when the integrations are performed.


### Local linear models
It is sometimes convenient to expand the architecture to include local linear models. In that case the architectures become, to first order,

 ${\displaystyle \varphi \left(\mathbf {x} \right)=\sum _{i=1}^{N}\left(a_{i}+\mathbf {b} _{i}\cdot \left(\mathbf {x} -\mathbf {c} _{i}\right)\right)\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

and

 ${\displaystyle \varphi \left(\mathbf {x} \right)=\sum _{i=1}^{N}\left(a_{i}+\mathbf {b} _{i}\cdot \left(\mathbf {x} -\mathbf {c} _{i}\right)\right)u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

in the unnormalized and normalized cases, respectively. Here  ${\displaystyle \mathbf {b} _{i}}$  are weights to be determined. Higher order linear terms are also possible.

This result can be written

 ${\displaystyle \varphi \left(\mathbf {x} \right)=\sum _{i=1}^{2N}\sum _{j=1}^{n}e_{ij}v_{ij}{\big (}\mathbf {x} -\mathbf {c} _{i}{\big )}}$ 

where

 ${\displaystyle e_{ij}={\begin{cases}a_{i},&{\mbox{if }}i\in [1,N]\\b_{ij},&{\mbox{if }}i\in [N+1,2N]\end{cases}}}$ 

and

 ${\displaystyle v_{ij}{\big (}\mathbf {x} -\mathbf {c} _{i}{\big )}\ {\stackrel {\mathrm {def} }{=}}\ {\begin{cases}\delta _{ij}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )},&{\mbox{if }}i\in [1,N]\\\left(x_{ij}-c_{ij}\right)\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )},&{\mbox{if }}i\in [N+1,2N]\end{cases}}}$ 

in the unnormalized case and in the normalized case.
Here  ${\displaystyle \delta _{ij}}$  is a Kronecker delta function defined as

 ${\displaystyle \delta _{ij}={\begin{cases}1,&{\mbox{if }}i=j\\0,&{\mbox{if }}i\neq j\end{cases}}}$ .


## Training
RBF networks are typically trained from pairs of input and target values  ${\displaystyle \mathbf {x} (t),y(t)}$ ,  ${\displaystyle t=1,\dots ,T}$  by a two-step algorithm.

In the first step, the center vectors  ${\displaystyle \mathbf {c} _{i}}$  of the RBF functions in the hidden layer are chosen. This step can be performed in several ways; centers can be randomly sampled from some set of examples, or they can be determined using k-means clustering. Note that this step is unsupervised.

The second step simply fits a linear model with coefficients  ${\displaystyle w_{i}}$  to the hidden layer's outputs with respect to some objective function. A common objective function, at least for regression/function estimation, is the least squares function:

 ${\displaystyle K(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ \sum _{t=1}^{T}K_{t}(\mathbf {w} )}$ 

where

 ${\displaystyle K_{t}(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}^{2}}$ .

We have explicitly included the dependence on the weights. Minimization of the least squares objective function by optimal choice of weights optimizes accuracy of fit.

There are occasions in which multiple objectives, such as smoothness as well as accuracy, must be optimized. In that case it is useful to optimize a regularized objective function such as

 ${\displaystyle H(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ K(\mathbf {w} )+\lambda S(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ \sum _{t=1}^{T}H_{t}(\mathbf {w} )}$ 

where

 ${\displaystyle S(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ \sum _{t=1}^{T}S_{t}(\mathbf {w} )}$ 

and

 ${\displaystyle H_{t}(\mathbf {w} )\ {\stackrel {\mathrm {def} }{=}}\ K_{t}(\mathbf {w} )+\lambda S_{t}(\mathbf {w} )}$ 

where optimization of S maximizes smoothness and  ${\displaystyle \lambda }$  is known as a regularization parameter.

A third optional backpropagation step can be performed to fine-tune all of the RBF net's parameters.


### Interpolation
RBF networks can be used to interpolate a function  ${\displaystyle y:\mathbb {R} ^{n}\to \mathbb {R} }$  when the values of that function are known on finite number of points:  ${\displaystyle y(\mathbf {x} _{i})=b_{i},i=1,\ldots ,N}$ . Taking the known points  ${\displaystyle \mathbf {x} _{i}}$  to be the centers of the radial basis functions and evaluating the values of the basis functions at the same points  ${\displaystyle g_{ij}=\rho (||\mathbf {x} _{j}-\mathbf {x} _{i}||)}$  the weights can be solved from the equation

 ${\displaystyle \left[{\begin{matrix}g_{11}&g_{12}&\cdots &g_{1N}\\g_{21}&g_{22}&\cdots &g_{2N}\\\vdots &&\ddots &\vdots \\g_{N1}&g_{N2}&\cdots &g_{NN}\end{matrix}}\right]\left[{\begin{matrix}w_{1}\\w_{2}\\\vdots \\w_{N}\end{matrix}}\right]=\left[{\begin{matrix}b_{1}\\b_{2}\\\vdots \\b_{N}\end{matrix}}\right]}$ 

It can be shown that the interpolation matrix in the above equation is non-singular, if the points  ${\displaystyle \mathbf {x} _{i}}$  are distinct, and thus the weights  ${\displaystyle w}$  can be solved by simple linear algebra:

 ${\displaystyle \mathbf {w} =\mathbf {G} ^{-1}\mathbf {b} }$ 

where  ${\displaystyle G=(g_{ij})}$ .


### Function approximation
If the purpose is not to perform strict interpolation but instead more general function approximation or classification the optimization is somewhat more complex because there is no obvious choice for the centers.  The training is typically done in two phases first fixing the width and centers and then the weights. This can be justified by considering the different nature of the non-linear hidden neurons versus the linear output neuron.


#### Training the basis function centers
Basis function centers can be randomly sampled among the input instances or obtained by Orthogonal Least Square Learning Algorithm or found by clustering the samples and choosing the cluster means as the centers.

The RBF widths are usually all fixed to same value which is proportional to the maximum distance between the chosen centers.


#### Pseudoinverse solution for the linear weights
After the centers  ${\displaystyle c_{i}}$  have been fixed, the weights that minimize the error at the output can be computed with a linear pseudoinverse solution:

 ${\displaystyle \mathbf {w} =\mathbf {G} ^{+}\mathbf {b} }$ ,

where the entries of G are the values of the radial basis functions evaluated at the points  ${\displaystyle x_{i}}$ :  ${\displaystyle g_{ji}=\rho (||x_{j}-c_{i}||)}$ .

The existence of this linear solution means that unlike multi-layer perceptron (MLP) networks, RBF networks have an explicit minimizer (when the centers are fixed).


#### Gradient descent training of the linear weights
Another possible training algorithm is gradient descent. In gradient descent training, the weights are adjusted at each time step by moving them in a direction opposite from the gradient of the objective function (thus allowing the minimum of the objective function to be found),

 ${\displaystyle \mathbf {w} (t+1)=\mathbf {w} (t)-\nu {\frac {d}{d\mathbf {w} }}H_{t}(\mathbf {w} )}$ 

where  ${\displaystyle \nu }$  is a "learning parameter."

For the case of training the linear weights,  ${\displaystyle a_{i}}$ , the algorithm becomes

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}\rho {\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}$ 

in the unnormalized case and

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}u{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}$ 

in the normalized case.

For local-linear-architectures gradient-descent training is

 ${\displaystyle e_{ij}(t+1)=e_{ij}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}v_{ij}{\big (}\mathbf {x} (t)-\mathbf {c} _{i}{\big )}}$ 


#### Projection operator training of the linear weights
For the case of training the linear weights,  ${\displaystyle a_{i}}$  and  ${\displaystyle e_{ij}}$ , the algorithm becomes

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}{\frac {\rho {\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho ^{2}{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}}}$ 

in the unnormalized case and

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}{\frac {u{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}u^{2}{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}}}$ 

in the normalized case and

 ${\displaystyle e_{ij}(t+1)=e_{ij}(t)+\nu {\big [}y(t)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}{\frac {v_{ij}{\big (}\mathbf {x} (t)-\mathbf {c} _{i}{\big )}}{\sum _{i=1}^{N}\sum _{j=1}^{n}v_{ij}^{2}{\big (}\mathbf {x} (t)-\mathbf {c} _{i}{\big )}}}}$ 

in the local-linear case.

For one basis function, projection operator training reduces to Newton's method.


## Examples

### Logistic map
The basic properties of radial basis functions can be illustrated with a simple mathematical map, the logistic map, which maps the unit interval onto itself. It can be used to generate a convenient prototype data stream. The logistic map can be used to explore function approximation, time series prediction, and control theory. The map originated from the field of population dynamics and became the prototype for chaotic time series. The map, in the fully chaotic regime, is given by

 ${\displaystyle x(t+1)\ {\stackrel {\mathrm {def} }{=}}\ f\left[x(t)\right]=4x(t)\left[1-x(t)\right]}$ 

where t is a time index. The value of x at time t+1 is a parabolic function of x at time t. This equation represents the underlying geometry of the chaotic time series generated by the logistic map.

Generation of the time series from this equation is the forward problem. The examples here illustrate the inverse problem; identification of the underlying dynamics, or fundamental equation, of the logistic map from exemplars of the time series. The goal is to find an estimate

 ${\displaystyle x(t+1)=f\left[x(t)\right]\approx \varphi (t)=\varphi \left[x(t)\right]}$ 

for f.


### Function approximation

#### Unnormalized radial basis functions
The architecture is

 ${\displaystyle \varphi (\mathbf {x} )\ {\stackrel {\mathrm {def} }{=}}\ \sum _{i=1}^{N}a_{i}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

where

 ${\displaystyle \rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}=\exp \left[-\beta _{i}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert ^{2}\right]=\exp \left[-\beta _{i}\left(x(t)-c_{i}\right)^{2}\right]}$ .

Since the input is a scalar rather than a vector, the input dimension is one. We choose the number of basis functions as N=5 and the size of the training set to be 100 exemplars generated by the chaotic time series. The weight  ${\displaystyle \beta }$  is taken to be a constant equal to 5. The weights  ${\displaystyle c_{i}}$  are five exemplars from the time series. The weights  ${\displaystyle a_{i}}$  are trained with projection operator training:

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}x(t+1)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}{\frac {\rho {\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho ^{2}{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}}}$ 

where the learning rate  ${\displaystyle \nu }$  is taken to be 0.3. The training is performed with one pass through the 100 training points. The rms error is 0.15.


#### Normalized radial basis functions
The normalized RBF architecture is

 ${\displaystyle \varphi (\mathbf {x} )\ {\stackrel {\mathrm {def} }{=}}\ {\frac {\sum _{i=1}^{N}a_{i}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}}=\sum _{i=1}^{N}a_{i}u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}$ 

where

 ${\displaystyle u{\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}\ {\stackrel {\mathrm {def} }{=}}\ {\frac {\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}\rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}}}}$ .

Again:

 ${\displaystyle \rho {\big (}\left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert {\big )}=\exp \left[-\beta \left\Vert \mathbf {x} -\mathbf {c} _{i}\right\Vert ^{2}\right]=\exp \left[-\beta \left(x(t)-c_{i}\right)^{2}\right]}$ .

Again, we choose the number of basis functions as five and the size of the training set to be 100 exemplars generated by the chaotic time series. The weight  ${\displaystyle \beta }$  is taken to be a constant equal to 6. The weights  ${\displaystyle c_{i}}$  are five exemplars from the time series. The weights  ${\displaystyle a_{i}}$  are trained with projection operator training:

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu {\big [}x(t+1)-\varphi {\big (}\mathbf {x} (t),\mathbf {w} {\big )}{\big ]}{\frac {u{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}u^{2}{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}}}$ 

where the learning rate  ${\displaystyle \nu }$  is again taken to be 0.3. The training is performed with one pass through the 100 training points. The rms error on a test set of 100 exemplars is 0.084, smaller than the unnormalized error. Normalization yields accuracy improvement. Typically accuracy with normalized basis functions increases even more over unnormalized functions as input dimensionality increases.


### Time series prediction
Once the underlying geometry of the time series is estimated as in the previous examples, a prediction for the time series can be made by iteration:

 ${\displaystyle \varphi (0)=x(1)}$ 

 ${\displaystyle {x}(t)\approx \varphi (t-1)}$ 

 ${\displaystyle {x}(t+1)\approx \varphi (t)=\varphi [\varphi (t-1)]}$ .

A comparison of the actual and estimated time series is displayed in the figure. The estimated times series starts out at time zero with an exact knowledge of x(0). It then uses the estimate  of the dynamics to update the time series estimate for several time steps.

Note that the estimate is accurate for only a few time steps. This is a general characteristic of chaotic time series. This is a property of the sensitive dependence on initial conditions common to chaotic time series. A small initial error is amplified with time. A measure of the divergence of time series with nearly identical initial conditions is known as the Lyapunov exponent.


### Control of a chaotic time series
We assume the output of the logistic map can be manipulated through a control parameter  ${\displaystyle c[x(t),t]}$  such that

 ${\displaystyle {x}_{}^{}(t+1)=4x(t)[1-x(t)]+c[x(t),t]}$ .

The goal is to choose the control parameter in such a way as to drive the time series to a desired output  ${\displaystyle d(t)}$ . This can be done if we choose the control parameter to be

 ${\displaystyle c_{}^{}[x(t),t]\ {\stackrel {\mathrm {def} }{=}}\ -\varphi [x(t)]+d(t+1)}$ 

where

 ${\displaystyle y[x(t)]\approx f[x(t)]=x(t+1)-c[x(t),t]}$ 

is an approximation to the underlying natural dynamics of the system.

The learning algorithm is given by

 ${\displaystyle a_{i}(t+1)=a_{i}(t)+\nu \varepsilon {\frac {u{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}{\sum _{i=1}^{N}u^{2}{\big (}\left\Vert \mathbf {x} (t)-\mathbf {c} _{i}\right\Vert {\big )}}}}$ 

where

 ${\displaystyle \varepsilon \ {\stackrel {\mathrm {def} }{=}}\ f[x(t)]-\varphi [x(t)]=x(t+1)-c[x(t),t]-\varphi [x(t)]=x(t+1)-d(t+1)}$ .