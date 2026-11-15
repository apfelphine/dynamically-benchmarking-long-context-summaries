# Variational autoencoder

## Overview of architecture and operation
A variational autoencoder is a generative model with a prior and noise distribution respectively. Usually such models are trained using the expectation-maximization meta-algorithm (e.g. probabilistic PCA, (spike & slab) sparse coding). Such a scheme optimizes a lower bound of the data likelihood, which is usually computationally intractable, and in doing so requires the discovery of q-distributions, or variational posteriors. These q-distributions are normally parameterized for each individual data point in a separate optimization process. However, variational autoencoders use a neural network as an amortized approach to jointly optimize across data points. In that way, the same parameters are reused for multiple data points, which can result in massive memory savings. The first neural network takes as input the data points themselves, and outputs parameters for the variational distribution. As it maps from a known input space to the low-dimensional latent space, it is called the encoder.

The decoder is the second neural network of this model. It is a function that maps from the latent space to the input space, e.g. as the means of the noise distribution. It is possible to use another neural network that maps to the variance, however this can be omitted for simplicity. In such a case, the variance can be optimized with gradient descent.

To optimize this model, one needs to know two terms: the "reconstruction error", and the Kullback–Leibler divergence (KL-D). Both terms are derived from the free energy expression of the probabilistic model, and therefore differ depending on the noise distribution and the assumed prior of the data, here referred to as p-distribution. For example, a standard VAE task such as IMAGENET is typically assumed to have a gaussianly distributed noise; however, tasks such as binarized MNIST require a Bernoulli noise. The KL-D from the free energy expression maximizes the probability mass of the q-distribution that overlaps with the p-distribution, which unfortunately can result in mode-seeking behaviour. The "reconstruction" term is the remainder of the free energy expression, and requires a sampling approximation to compute its expectation value.

More recent approaches replace Kullback–Leibler divergence (KL-D) with various statistical distances, see "Statistical distance VAE variants" below.


## Formulation
From the point of view of probabilistic modeling, one wants to maximize the likelihood of the data  ${\displaystyle x}$  by their chosen parameterized probability distribution  ${\displaystyle p_{\theta }(x)=p(x|\theta )}$ . This distribution is usually chosen to be a Gaussian  ${\displaystyle N(x|\mu ,\sigma )}$  which is parameterized by  ${\displaystyle \mu }$  and  ${\displaystyle \sigma }$  respectively, and as a member of the exponential family it is easy to work with as a noise distribution. Simple distributions are easy enough to maximize, however distributions where a prior is assumed over the latents  ${\displaystyle z}$  results in intractable integrals. Let us find  ${\displaystyle p_{\theta }(x)}$  via marginalizing over  ${\displaystyle z}$ .

 ${\displaystyle p_{\theta }(x)=\int _{z}p_{\theta }({x,z})\,dz,}$ 

where  ${\displaystyle p_{\theta }({x,z})}$  represents the joint distribution under  ${\displaystyle p_{\theta }}$  of the observable data  ${\displaystyle x}$  and its latent representation or encoding  ${\displaystyle z}$ . According to the chain rule, the equation can be rewritten as

 ${\displaystyle p_{\theta }(x)=\int _{z}p_{\theta }({x|z})p_{\theta }(z)\,dz}$ 

In the vanilla variational autoencoder,  ${\displaystyle z}$  is usually taken to be a finite-dimensional vector of real numbers, and  ${\displaystyle p_{\theta }({x|z})}$  to be a Gaussian distribution. Then  ${\displaystyle p_{\theta }(x)}$  is a mixture of Gaussian distributions.

It is now possible to define the set of the relationships between the input data and its latent representation as

- Prior  ${\displaystyle p_{\theta }(z)}$
- Likelihood  ${\displaystyle p_{\theta }(x|z)}$
- Posterior  ${\displaystyle p_{\theta }(z|x)}$


Unfortunately, the computation of  ${\displaystyle p_{\theta }(z|x)}$  is expensive and in most cases intractable. To speed up the calculus to make it feasible, it is necessary to introduce a further function to approximate the posterior distribution as

 ${\displaystyle q_{\phi }({z|x})\approx p_{\theta }({z|x})}$ 

with  ${\displaystyle \phi }$  defined as the set of real values that parametrize  ${\displaystyle q}$ . This is sometimes called amortized inference, since by "investing" in finding a good  ${\displaystyle q_{\phi }}$ , one can later infer  ${\displaystyle z}$  from  ${\displaystyle x}$  quickly without doing any integrals.

In this way, the problem is to find a good probabilistic autoencoder, in which the conditional likelihood distribution  ${\displaystyle p_{\theta }(x|z)}$  is computed by the probabilistic decoder, and the approximated posterior distribution  ${\displaystyle q_{\phi }(z|x)}$  is computed by the probabilistic encoder.

Parametrize the encoder as  ${\displaystyle E_{\phi }}$ , and the decoder as  ${\displaystyle D_{\theta }}$ .


## Evidence lower bound (ELBO)
Like many deep learning approaches that use gradient-based optimization, VAEs require a differentiable loss function to update the network weights through backpropagation.

For variational autoencoders, the idea is to jointly optimize the generative model parameters  ${\displaystyle \theta }$  to reduce the reconstruction error between the input and the output, and  ${\displaystyle \phi }$  to make  ${\displaystyle q_{\phi }({z|x})}$  as close as possible to  ${\displaystyle p_{\theta }(z|x)}$ . As reconstruction loss, mean squared error and cross entropy are often used.

The Kullback–Leibler divergence  ${\displaystyle D_{KL}(q_{\phi }({z|x})\parallel p_{\theta }({z|x}))}$  can be used as a loss function to squeeze  ${\displaystyle q_{\phi }({z|x})}$  under  ${\displaystyle p_{\theta }(z|x)}$ . This divergence loss expands to

 ${\displaystyle {\begin{aligned}D_{KL}(q_{\phi }({z|x})\parallel p_{\theta }({z|x}))&=\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {q_{\phi }(z|x)}{p_{\theta }(z|x)}}\right]\\&=\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {q_{\phi }({z|x})p_{\theta }(x)}{p_{\theta }(x,z)}}\right]\\&=\ln p_{\theta }(x)+\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {q_{\phi }({z|x})}{p_{\theta }(x,z)}}\right].\end{aligned}}}$ 

Now, define the evidence lower bound (ELBO): ${\displaystyle L_{\theta ,\phi }(x):=\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {p_{\theta }(x,z)}{q_{\phi }({z|x})}}\right]=\ln p_{\theta }(x)-D_{KL}(q_{\phi }({\cdot |x})\parallel p_{\theta }({\cdot |x}))}$ Maximizing the ELBO ${\displaystyle \theta ^{*},\phi ^{*}={\underset {\theta ,\phi }{\operatorname {argmax} }}\,L_{\theta ,\phi }(x)}$ is equivalent to simultaneously maximizing  ${\displaystyle \ln p_{\theta }(x)}$  and minimizing  ${\displaystyle D_{KL}(q_{\phi }({z|x})\parallel p_{\theta }({z|x}))}$ . That is, maximizing the log-likelihood of the observed data, and minimizing the divergence from the approximate posterior  ${\displaystyle q_{\phi }(\cdot |x)}$  to the exact posterior  ${\displaystyle p_{\theta }(\cdot |x)}$ .

The form given is not very convenient for maximization, but the following, equivalent form, is: ${\displaystyle L_{\theta ,\phi }(x)=\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln p_{\theta }(x|z)\right]-D_{KL}(q_{\phi }({\cdot |x})\parallel p_{\theta }(\cdot ))}$ where  ${\displaystyle \ln p_{\theta }(x|z)}$  is implemented as  ${\displaystyle -{\frac {1}{2}}\|x-D_{\theta }(z)\|_{2}^{2}}$ , since that is, up to an additive constant, what  ${\displaystyle x|z\sim {\mathcal {N}}(D_{\theta }(z),I)}$  yields. That is, we model the distribution of  ${\displaystyle x}$  conditional on  ${\displaystyle z}$  to be a Gaussian distribution centered on  ${\displaystyle D_{\theta }(z)}$ . The distribution of  ${\displaystyle q_{\phi }(z|x)}$  and  ${\displaystyle p_{\theta }(z)}$  are often also chosen to be Gaussians as  ${\displaystyle z|x\sim {\mathcal {N}}(E_{\phi }(x),\sigma _{\phi }(x)^{2}I)}$  and  ${\displaystyle z\sim {\mathcal {N}}(0,I)}$ , with which we obtain by the formula for KL divergence of Gaussians: ${\displaystyle L_{\theta ,\phi }(x)=-{\frac {1}{2}}\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\|x-D_{\theta }(z)\|_{2}^{2}\right]-{\frac {1}{2}}\left(N\sigma _{\phi }(x)^{2}+\|E_{\phi }(x)\|_{2}^{2}-2N\ln \sigma _{\phi }(x)\right)+Const}$ Here  ${\displaystyle N}$  is the dimension of  ${\displaystyle z}$ . For a more detailed derivation and more interpretations of ELBO and its maximization, see its main page.


## Reparameterization
To efficiently search for  ${\displaystyle \theta ^{*},\phi ^{*}={\underset {\theta ,\phi }{\operatorname {argmax} }}\,L_{\theta ,\phi }(x)}$ the typical method is gradient ascent.

It is straightforward to find ${\displaystyle \nabla _{\theta }\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {p_{\theta }(x,z)}{q_{\phi }({z|x})}}\right]=\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\nabla _{\theta }\ln {\frac {p_{\theta }(x,z)}{q_{\phi }({z|x})}}\right]}$ However,  ${\displaystyle \nabla _{\phi }\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {p_{\theta }(x,z)}{q_{\phi }({z|x})}}\right]}$ does not allow one to put the  ${\displaystyle \nabla _{\phi }}$  inside the expectation, since  ${\displaystyle \phi }$  appears in the probability distribution itself. The reparameterization trick (also known as stochastic backpropagation) bypasses this difficulty.

The most important example is when  ${\displaystyle z\sim q_{\phi }(\cdot |x)}$  is normally distributed, as  ${\displaystyle {\mathcal {N}}(\mu _{\phi }(x),\Sigma _{\phi }(x))}$ .

This can be reparametrized by letting  ${\displaystyle {\boldsymbol {\varepsilon }}\sim {\mathcal {N}}(0,{\boldsymbol {I}})}$  be a "standard random number generator", and construct  ${\displaystyle z}$  as  ${\displaystyle z=\mu _{\phi }(x)+L_{\phi }(x)\epsilon }$ . Here,  ${\displaystyle L_{\phi }(x)}$  is obtained by the Cholesky decomposition: ${\displaystyle \Sigma _{\phi }(x)=L_{\phi }(x)L_{\phi }(x)^{T}}$ Then we have ${\displaystyle \nabla _{\phi }\mathbb {E} _{z\sim q_{\phi }(\cdot |x)}\left[\ln {\frac {p_{\theta }(x,z)}{q_{\phi }({z|x})}}\right]=\mathbb {E} _{\epsilon }\left[\nabla _{\phi }\ln {\frac {p_{\theta }(x,\mu _{\phi }(x)+L_{\phi }(x)\epsilon )}{q_{\phi }(\mu _{\phi }(x)+L_{\phi }(x)\epsilon |x)}}\right]}$ and so we obtained an unbiased estimator of the gradient, allowing stochastic gradient descent.

Since we reparametrized  ${\displaystyle z}$ , we need to find  ${\displaystyle q_{\phi }(z|x)}$ . Let  ${\displaystyle q_{0}}$  be the probability density function for  ${\displaystyle \epsilon }$ , then  ${\displaystyle \ln q_{\phi }(z|x)=\ln q_{0}(\epsilon )-\ln |\det(\partial _{\epsilon }z)|}$ where  ${\displaystyle \partial _{\epsilon }z}$  is the Jacobian matrix of  ${\displaystyle z}$  with respect to  ${\displaystyle \epsilon }$ . Since  ${\displaystyle z=\mu _{\phi }(x)+L_{\phi }(x)\epsilon }$ , this is  ${\displaystyle \ln q_{\phi }(z|x)=-{\frac {1}{2}}\|\epsilon \|^{2}-\ln |\det L_{\phi }(x)|-{\frac {n}{2}}\ln(2\pi )}$ 


## Variations
Many variational autoencoders applications and extensions have been used to adapt the architecture to other domains and improve its performance. 

 ${\displaystyle \beta }$ -VAE is an implementation with a weighted Kullback–Leibler divergence term to automatically discover and interpret factorised latent representations. With this implementation, it is possible to force manifold disentanglement for  ${\displaystyle \beta }$  values greater than one. This architecture can discover disentangled latent factors without supervision.

The conditional VAE (CVAE), inserts label information in the latent space to force a deterministic constrained representation of the learned data.

Some structures directly deal with the quality of the generated samples or implement more than one latent space to further improve the representation learning.

Some architectures mix VAE and generative adversarial networks to obtain hybrid models.

It is not necessary to use gradients to update the encoder. In fact, the encoder is not necessary for the generative model. 


## Statistical distance VAE variants
After the initial work of Diederik P. Kingma and Max Welling, several procedures were 
proposed to formulate in a more abstract way the operation of the VAE. In these approaches the loss function is composed of two parts : 

- the usual reconstruction error part which seeks to ensure that the encoder-then-decoder mapping  ${\displaystyle x\mapsto D_{\theta }(E_{\psi }(x))}$  is as close to the identity map as possible; the sampling is done at run time from the empirical distribution   ${\displaystyle \mathbb {P} ^{real}}$  of objects available (e.g., for MNIST or IMAGENET this will be the empirical probability law of all images in the dataset). This gives the term:   ${\displaystyle \mathbb {E} _{x\sim \mathbb {P} ^{real}}\left[\|x-D_{\theta }(E_{\phi }(x))\|_{2}^{2}\right]}$ .
- a variational part that ensures that, when the empirical distribution  ${\displaystyle \mathbb {P} ^{real}}$  is passed through the encoder  ${\displaystyle E_{\phi }}$ , we recover the target distribution, denoted here  ${\displaystyle \mu (dz)}$  that is usually taken to be a Multivariate normal distribution. We will denote  ${\displaystyle E_{\phi }\sharp \mathbb {P} ^{real}}$  this pushforward measure which in practice is just the empirical distribution obtained by passing all dataset objects through the encoder  ${\displaystyle E_{\phi }}$ . In order to make sure that   ${\displaystyle E_{\phi }\sharp \mathbb {P} ^{real}}$   is close to the target  ${\displaystyle \mu (dz)}$ , a Statistical distance  ${\displaystyle d}$  is invoked and the term   ${\displaystyle d\left(\mu (dz),E_{\phi }\sharp \mathbb {P} ^{real}\right)^{2}}$  is added to the loss.


We obtain the final formula for the loss:
 ${\displaystyle L_{\theta ,\phi }=\mathbb {E} _{x\sim \mathbb {P} ^{real}}\left[\|x-D_{\theta }(E_{\phi }(x))\|_{2}^{2}\right]+d\left(\mu (dz),E_{\phi }\sharp \mathbb {P} ^{real}\right)^{2}}$ 

The statistical distance  ${\displaystyle d}$  requires special properties, for instance it has to be posses a formula as expectation because the loss function will need to be optimized by stochastic optimization algorithms. Several distances can be chosen and this gave rise to several flavors of VAEs:

- the sliced Wasserstein distance used by S Kolouri, et al. in their VAE[23]
- the energy distance implemented in the Radon Sobolev Variational Auto-Encoder[24]
- the Maximum Mean Discrepancy distance used in the MMD-VAE[25]
- the Wasserstein distance used in the WAEs[26]
- kernel-based distances used in the Kernelized Variational Autoencoder (K-VAE)[27]