# Wasserstein GAN

## Motivation

### The GAN game
The original GAN method is based on the GAN game, a zero-sum game with 2 players: generator and discriminator. The game is defined over a probability space  ${\displaystyle (\Omega ,{\mathcal {B}},\mu _{ref})}$ , The generator's strategy set is the set of all probability measures  ${\displaystyle \mu _{G}}$  on  ${\displaystyle (\Omega ,{\mathcal {B}})}$ , and the discriminator's strategy set is the set of measurable functions  ${\displaystyle D:\Omega \to [0,1]}$ .

The objective of the game is ${\displaystyle L(\mu _{G},D):=\mathbb {E} _{x\sim \mu _{ref}}[\ln D(x)]+\mathbb {E} _{x\sim \mu _{G}}[\ln(1-D(x))].}$ 
The generator aims to minimize it, and the discriminator aims to maximize it.



A basic theorem of the GAN game states that

Theorem (the optimal discriminator computes the Jensen–Shannon divergence)—For any fixed generator strategy  ${\displaystyle \mu _{G}}$ , let the optimal reply be  ${\displaystyle D^{*}=\arg \max _{D}L(\mu _{G},D)}$ , then

 ${\displaystyle {\begin{aligned}D^{*}(x)&={\frac {d\mu _{ref}}{d(\mu _{ref}+\mu _{G})}}\\L(\mu _{G},D^{*})&=2D_{JS}(\mu _{ref};\mu _{G})-2\ln 2,\end{aligned}}}$ 

where the derivative is the Radon–Nikodym derivative, and  ${\displaystyle D_{JS}}$  is the Jensen–Shannon divergence.

Repeat the GAN game many times, each time with the generator moving first, and the discriminator moving second. Each time the generator  ${\displaystyle \mu _{G}}$  changes, the discriminator must adapt by approaching the ideal ${\displaystyle D^{*}(x)={\frac {d\mu _{ref}}{d(\mu _{ref}+\mu _{G})}}.}$ 
Since we are really interested in  ${\displaystyle \mu _{ref}}$ , the discriminator function  ${\displaystyle D}$  is by itself rather uninteresting. It merely keeps track of the likelihood ratio between the generator distribution and the reference distribution. At equilibrium, the discriminator is just outputting  ${\displaystyle {\frac {1}{2}}}$  constantly, having given up trying to perceive any difference.

Concretely, in the GAN game, let us fix a generator  ${\displaystyle \mu _{G}}$ , and improve the discriminator step-by-step, with  ${\displaystyle \mu _{D,t}}$  being the discriminator at step  ${\displaystyle t}$ . Then we (ideally) have ${\displaystyle L(\mu _{G},\mu _{D,1})\leq L(\mu _{G},\mu _{D,2})\leq \cdots \leq \max _{\mu _{D}}L(\mu _{G},\mu _{D})=2D_{JS}(\mu _{ref}\|\mu _{G})-2\ln 2,}$ so we see that the discriminator is actually lower-bounding  ${\displaystyle D_{JS}(\mu _{ref}\|\mu _{G})}$ .


### Wasserstein distance
Thus, we see that the point of the discriminator is mainly as a critic to provide feedback for the generator, about "how far it is from perfection", where "far" is defined as Jensen–Shannon divergence.

Naturally, this brings the possibility of using a different criteria of farness. There are many possible divergences to choose from, such as the f-divergence family, which would give the f-GAN.

The Wasserstein GAN is obtained by using the Wasserstein metric, which satisfies a "dual representation theorem" that renders it highly efficient to compute:

Theorem (Kantorovich-Rubenstein duality)—When the probability space  ${\displaystyle \Omega }$  is a metric space, then  
for any fixed  ${\displaystyle K>0}$ ,  ${\displaystyle W_{1}(\mu ,\nu )={\frac {1}{K}}\sup _{\|f\|_{L}\leq K}\mathbb {E} _{x\sim \mu }[f(x)]-\mathbb {E} _{y\sim \nu }[f(y)]}$ 
where  ${\displaystyle \|\cdot \|_{L}}$  is the Lipschitz norm.

A proof can be found in the main page on Wasserstein metric.


## Definition
By the Kantorovich-Rubenstein duality, the definition of Wasserstein GAN is clear:

A Wasserstein GAN game is defined by a probability space  ${\displaystyle (\Omega ,{\mathcal {B}},\mu _{ref})}$ , where  ${\displaystyle \Omega }$  is a metric space, and a constant  ${\displaystyle K>0}$ .
There are 2 players: generator and discriminator (also called "critic").
The generator's strategy set is the set of all probability measures  ${\displaystyle \mu _{G}}$  on  ${\displaystyle (\Omega ,{\mathcal {B}})}$ .
The discriminator's strategy set is the set of measurable functions of type  ${\displaystyle D:\Omega \to \mathbb {R} }$  with bounded Lipschitz-norm:  ${\displaystyle \|D\|_{L}\leq K}$ .
The Wasserstein GAN game is a zero-sum game, with objective function ${\displaystyle L_{WGAN}(\mu _{G},D):=\mathbb {E} _{x\sim \mu _{G}}[D(x)]-\mathbb {E} _{x\sim \mu _{ref}}[D(x)].}$ 
The generator goes first, and the discriminator goes second. The generator aims to minimize the objective, and the discriminator aims to maximize the objective: ${\displaystyle \min _{\mu _{G}}\max _{D}L_{WGAN}(\mu _{G},D).}$ 

By the Kantorovich-Rubenstein duality, for any generator strategy  ${\displaystyle \mu _{G}}$ , the optimal reply by the discriminator is  ${\displaystyle D^{*}}$ , such that  ${\displaystyle L_{WGAN}(\mu _{G},D^{*})=K\cdot W_{1}(\mu _{G},\mu _{ref}).}$ Consequently, if the discriminator is good, the generator would be constantly pushed to minimize  ${\displaystyle W_{1}(\mu _{G},\mu _{ref})}$ , and the optimal strategy for the generator is just  ${\displaystyle \mu _{G}=\mu _{ref}}$ , as it should.


## Comparison with GAN
In the Wasserstein GAN game, the discriminator provides a better gradient than in the GAN game.

Consider for example a game on the real line where both  ${\displaystyle \mu _{G}}$  and  ${\displaystyle \mu _{ref}}$  are Gaussian. Then the optimal Wasserstein critic  ${\displaystyle D_{WGAN}}$  and the optimal GAN discriminator  ${\displaystyle D}$  are plotted as below:

For fixed discriminator, the generator needs to minimize the following objectives:

- For GAN,  ${\displaystyle \mathbb {E} _{x\sim \mu _{G}}[\ln(1-D(x))]}$ .
- For Wasserstein GAN,  ${\displaystyle \mathbb {E} _{x\sim \mu _{G}}[D_{WGAN}(x)]}$ .


Let  ${\displaystyle \mu _{G}}$  be parametrized by  ${\displaystyle \theta }$ , then we can perform stochastic gradient descent by using two unbiased estimators of the gradient: ${\displaystyle \nabla _{\theta }\mathbb {E} _{x\sim \mu _{G}}[\ln(1-D(x))]=\mathbb {E} _{x\sim \mu _{G}}[\ln(1-D(x))\cdot \nabla _{\theta }\ln \rho _{\mu _{G}}(x)]}$  ${\displaystyle \nabla _{\theta }\mathbb {E} _{x\sim \mu _{G}}[D_{WGAN}(x)]=\mathbb {E} _{x\sim \mu _{G}}[D_{WGAN}(x)\cdot \nabla _{\theta }\ln \rho _{\mu _{G}}(x)]}$ where we used the reparameterization trick.

As shown, the generator in GAN is motivated to let its  ${\displaystyle \mu _{G}}$  "slide down the peak" of  ${\displaystyle \ln(1-D(x))}$ . Similarly for the generator in Wasserstein GAN.

For Wasserstein GAN,  ${\displaystyle D_{WGAN}}$  has gradient 1 almost everywhere, while for GAN,  ${\displaystyle \ln(1-D)}$  has flat gradient in the middle, and steep gradient elsewhere. As a result, the variance for the estimator in GAN is usually much larger than that in Wasserstein GAN. See also Figure 3 of.

The problem with  ${\displaystyle D_{JS}}$  is much more severe in actual machine learning situations. Consider training a GAN to generate ImageNet, a collection of photos of size 256-by-256. The space of all such photos is  ${\displaystyle \mathbb {R} ^{256^{2}}}$ , and the distribution of ImageNet pictures,  ${\displaystyle \mu _{ref}}$ , concentrates on a manifold of much lower dimension in it. Consequently, any generator strategy  ${\displaystyle \mu _{G}}$  would almost surely be entirely disjoint from  ${\displaystyle \mu _{ref}}$ , making  ${\displaystyle D_{JS}(\mu _{G}\|\mu _{ref})=+\infty }$ . Thus, a good discriminator can almost perfectly distinguish  ${\displaystyle \mu _{ref}}$  from  ${\displaystyle \mu _{G}}$ , as well as any  ${\displaystyle \mu _{G}'}$  close to  ${\displaystyle \mu _{G}}$ . Thus, the gradient  ${\displaystyle \nabla _{\mu _{G}}L(\mu _{G},D)\approx 0}$ , creating no learning signal for the generator.

Detailed theorems can be found in.


## Training Wasserstein GANs
Training the generator in Wasserstein GAN is just gradient descent, the same as in GAN (or most deep learning methods), but training the discriminator is different, as the discriminator is now restricted to have bounded Lipschitz norm. There are several methods for this.


### Upper-bounding the Lipschitz norm
Let the discriminator function  ${\displaystyle D}$  to be implemented by a multilayer perceptron: ${\displaystyle D=D_{n}\circ D_{n-1}\circ \cdots \circ D_{1}}$ where  ${\displaystyle D_{i}(x)=h(W_{i}x)}$ , and  ${\displaystyle h:\mathbb {R} \to \mathbb {R} }$  is a fixed activation function with  ${\displaystyle \sup _{x}|h'(x)|\leq 1}$ . For example, the hyperbolic tangent function  ${\displaystyle h=\tanh }$  satisfies the requirement.

Then, for any  ${\displaystyle x}$ , let  ${\displaystyle x_{i}=(D_{i}\circ D_{i-1}\circ \cdots \circ D_{1})(x)}$ , we have by the chain rule: ${\displaystyle dD(x)=diag(h'(W_{n}x_{n-1}))\cdot W_{n}\cdot diag(h'(W_{n-1}x_{n-2}))\cdot W_{n-1}\cdots diag(h'(W_{1}x))\cdot W_{1}\cdot dx}$ Thus, the Lipschitz norm of  ${\displaystyle D}$  is upper-bounded by ${\displaystyle \|D\|_{L}\leq \sup _{x}\|diag(h'(W_{n}x_{n-1}))\cdot W_{n}\cdot diag(h'(W_{n-1}x_{n-2}))\cdot W_{n-1}\cdots diag(h'(W_{1}x))\cdot W_{1}\|_{F}}$ where  ${\displaystyle \|\cdot \|_{s}}$  is the operator norm of the matrix, that is, the largest singular value of the matrix, that is, the spectral radius of the matrix (these concepts are the same for matrices, but different for general linear operators).

Since  ${\displaystyle \sup _{x}|h'(x)|\leq 1}$ , we have  ${\displaystyle \|diag(h'(W_{i}x_{i-1}))\|_{s}=\max _{j}|h'(W_{i}x_{i-1,j})|\leq 1}$ , and consequently the upper bound: ${\displaystyle \|D\|_{L}\leq \prod _{i=1}^{n}\|W_{i}\|_{s}}$ Thus, if we can upper-bound operator norms  ${\displaystyle \|W_{i}\|_{s}}$  of each matrix, we can upper-bound the Lipschitz norm of  ${\displaystyle D}$ .


### Weight clipping
Since for any  ${\displaystyle m\times l}$  matrix  ${\displaystyle W}$ , let  ${\displaystyle c=\max _{i,j}|W_{i,j}|}$ , we have ${\displaystyle \|W\|_{s}^{2}=\sup _{\|x\|_{2}=1}\|Wx\|_{2}^{2}=\sup _{\|x\|_{2}=1}\sum _{i}\left(\sum _{j}W_{i,j}x_{j}\right)^{2}=\sup _{\|x\|_{2}=1}\sum _{i,j,k}W_{ij}W_{ik}x_{j}x_{k}\leq c^{2}ml^{2}}$ by clipping all entries of  ${\displaystyle W}$  to within some interval  ${\displaystyle [-c,c]}$ , we have can bound  ${\displaystyle \|W\|_{s}}$ .

This is the weight clipping method, proposed by the original paper.


### Spectral normalization
The spectral radius can be efficiently computed by the following algorithm:


INPUT matrix  ${\displaystyle W}$  and initial guess  ${\displaystyle x}$ 
Iterate  ${\displaystyle x\mapsto {\frac {1}{\|Wx\|_{2}}}Wx}$  to convergence  ${\displaystyle x^{*}}$ . This is the eigenvector of  ${\displaystyle W}$  with eigenvalue  ${\displaystyle \|W\|_{s}}$ .
RETURN  ${\displaystyle x^{*},\|Wx^{*}\|_{2}}$ 

By reassigning  ${\displaystyle W_{i}\leftarrow {\frac {W_{i}}{\|W_{i}\|_{s}}}}$  after each update of the discriminator, we can upper bound  ${\displaystyle \|W_{i}\|_{s}\leq 1}$ , and thus upper bound  ${\displaystyle \|D\|_{L}}$ .

The algorithm can be further accelerated by memoization: At step  ${\displaystyle t}$ , store  ${\displaystyle x_{i}^{*}(t)}$ . Then at step  ${\displaystyle t+1}$ , use  ${\displaystyle x_{i}^{*}(t)}$  as the initial guess for the algorithm. Since  ${\displaystyle W_{i}(t+1)}$  is very close to  ${\displaystyle W_{i}(t)}$ , so is  ${\displaystyle x_{i}^{*}(t)}$  close to  ${\displaystyle x_{i}^{*}(t+1)}$ , so this allows rapid convergence.

This is the spectral normalization method.


### Gradient penalty
Instead of strictly bounding  ${\displaystyle \|D\|_{L}}$ , we can simply add a "gradient penalty" term for the discriminator, of form ${\displaystyle \mathbb {E} _{x\sim {\hat {\mu }}}[(\|\nabla D(x)\|_{2}-a)^{2}]}$ where  ${\displaystyle {\hat {\mu }}}$  is a fixed distribution used to estimate how much the discriminator has violated the Lipschitz norm requirement.
The discriminator, in attempting to minimize the new loss function, would naturally bring  ${\displaystyle \nabla D(x)}$  close to  ${\displaystyle a}$  everywhere, thus making  ${\displaystyle \|D\|_{L}\approx a}$ .

This is the gradient penalty method.