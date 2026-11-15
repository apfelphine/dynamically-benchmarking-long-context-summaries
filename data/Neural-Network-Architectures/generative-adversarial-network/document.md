# Generative adversarial network

## Definition

### Mathematical
The original GAN is defined as the following game:


Each probability space  ${\displaystyle (\Omega ,\mu _{\text{ref}})}$  defines a GAN game.
There are 2 players: generator and discriminator.
The generator's strategy set is  ${\displaystyle {\mathcal {P}}(\Omega )}$ , the set of all probability measures  ${\displaystyle \mu _{G}}$  on  ${\displaystyle \Omega }$ .
The discriminator's strategy set is the set of Markov kernels  ${\displaystyle \mu _{D}:\Omega \to {\mathcal {P}}[0,1]}$ , where  ${\displaystyle {\mathcal {P}}[0,1]}$  is the set of probability measures on  ${\displaystyle [0,1]}$ .
The GAN game is a zero-sum game, with objective function ${\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\operatorname {E} _{x\sim \mu _{G},y\sim \mu _{D}(x)}[\ln(1-y)].}$ 
The generator aims to minimize the objective, and the discriminator aims to maximize the objective.


The generator's task is to approach  ${\displaystyle \mu _{G}\approx \mu _{\text{ref}}}$ , that is, to match its own output distribution as closely as possible to the reference distribution. The discriminator's task is to output a value close to 1 when the input appears to be from the reference distribution, and to output a value close to 0 when the input looks like it came from the generator distribution.


### In practice
The generative network generates candidates while the discriminative network evaluates them. This creates a contest based on data distributions, where the generator learns to map from a latent space to the true data distribution, aiming to produce candidates that the discriminator cannot distinguish from real data. The discriminator's goal is to correctly identify these candidates, but as the generator improves, its task becomes more challenging, increasing the discriminator's error rate.

A known dataset serves as the initial training data for the discriminator. Training involves presenting it with samples from the training dataset until it achieves acceptable accuracy. The generator is trained based on whether it succeeds in fooling the discriminator. Typically, the generator is seeded with randomized input that is sampled from a predefined latent space (e.g. a multivariate normal distribution). Thereafter, candidates synthesized by the generator are evaluated by the discriminator. Independent backpropagation procedures are applied to both networks so that the generator produces better samples, while the discriminator becomes more skilled at flagging synthetic samples. When used for image generation, the generator is typically a deconvolutional neural network, and the discriminator is a convolutional neural network.


### Relation to other statistical machine learning methods
GANs are implicit generative models, which means that they do not explicitly model the likelihood function nor provide a means for finding the latent variable corresponding to a given sample, unlike alternatives such as flow-based generative model.

Compared to fully visible belief networks such as WaveNet and PixelRNN and autoregressive models in general, GANs can generate one complete sample in one pass, rather than multiple passes through the network.

Compared to Boltzmann machines and linear ICA, there is no restriction on the type of function used by the network.

Since neural networks are universal approximators, GANs are asymptotically consistent. Variational autoencoders might be universal approximators, but it is not proven as of 2017.


## Mathematical properties

### Measure-theoretic considerations
This section provides some of the mathematical theory behind these methods.



In modern probability theory based on measure theory, a probability space also needs to be equipped with a σ-algebra. As a result, a more rigorous definition of the GAN game would make the following changes:

Each probability space  ${\displaystyle (\Omega ,{\mathcal {B}},\mu _{\text{ref}})}$  defines a GAN game.
The generator's strategy set is  ${\displaystyle {\mathcal {P}}(\Omega ,{\mathcal {B}})}$ , the set of all probability measures  ${\displaystyle \mu _{G}}$  on the measure-space  ${\displaystyle (\Omega ,{\mathcal {B}})}$ .

The discriminator's strategy set is the set of Markov kernels  ${\displaystyle \mu _{D}:(\Omega ,{\mathcal {B}})\to {\mathcal {P}}([0,1],{\mathcal {B}}([0,1]))}$ , where  ${\displaystyle {\mathcal {B}}([0,1])}$  is the Borel σ-algebra on  ${\displaystyle [0,1]}$ .

Since issues of measurability never arise in practice, these will not concern us further.


### Choice of the strategy set
In the most generic version of the GAN game described above, the strategy set for the discriminator contains all Markov kernels  ${\displaystyle \mu _{D}:\Omega \to {\mathcal {P}}[0,1]}$ , and the strategy set for the generator contains arbitrary probability distributions  ${\displaystyle \mu _{G}}$  on  ${\displaystyle \Omega }$ .

However, as shown below, the optimal discriminator strategy against any  ${\displaystyle \mu _{G}}$  is deterministic, so there is no loss of generality in restricting the discriminator's strategies to deterministic functions  ${\displaystyle D:\Omega \to [0,1]}$ . In most applications,  ${\displaystyle D}$  is a deep neural network function.

As for the generator, while  ${\displaystyle \mu _{G}}$  could theoretically be any computable probability distribution, in practice, it is usually implemented as a pushforward:  ${\displaystyle \mu _{G}=\mu _{Z}\circ G^{-1}}$ . That is, start with a random variable  ${\displaystyle z\sim \mu _{Z}}$ , where  ${\displaystyle \mu _{Z}}$  is a probability distribution that is easy to compute (such as the uniform distribution, or the Gaussian distribution), then define a function  ${\displaystyle G:\Omega _{Z}\to \Omega }$ . Then the distribution  ${\displaystyle \mu _{G}}$  is the distribution of  ${\displaystyle G(z)}$ .

Consequently, the generator's strategy is usually defined as just  ${\displaystyle G}$ , leaving  ${\displaystyle z\sim \mu _{Z}}$  implicit. In this formalism, the GAN game objective is ${\displaystyle L(G,D):=\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\operatorname {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z)))].}$ 


### Generative reparametrization
The GAN architecture has two main components. One is casting optimization into a game, of form  ${\displaystyle \min _{G}\max _{D}L(G,D)}$ , which is different from the usual kind of optimization, of form  ${\displaystyle \min _{\theta }L(\theta )}$ . The other is the decomposition of  ${\displaystyle \mu _{G}}$  into  ${\displaystyle \mu _{Z}\circ G^{-1}}$ , which can be understood as a reparametrization trick.

To see its significance, one must compare GAN with previous methods for learning generative models, which were plagued with "intractable probabilistic computations that arise in maximum likelihood estimation and related strategies".

At the same time, Kingma and Welling and Rezende et al. developed the same idea of reparametrization into a general stochastic backpropagation method. Among its first applications was the variational autoencoder.


### Move order and strategic equilibria
In the original paper, as well as most subsequent papers, it is usually assumed that the generator moves first, and the discriminator moves second, thus giving the following minimax game: ${\displaystyle \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\operatorname {E} _{x\sim \mu _{G},y\sim \mu _{D}(x)}[\ln(1-y)].}$ 

If both the generator's and the discriminator's strategy sets are spanned by a finite number of strategies, then by the minimax theorem, ${\displaystyle \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=\max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})}$ that is, the move order does not matter.

However, since the strategy sets are both not finitely spanned, the minimax theorem does not apply, and the idea of an "equilibrium" becomes delicate. To wit, there are the following different concepts of equilibrium:

- Equilibrium when generator moves first, and discriminator moves second: ${\displaystyle {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D}),\quad {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),\quad }$
- Equilibrium when discriminator moves first, and generator moves second: ${\displaystyle {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D}),\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D}),}$
- Nash equilibrium  ${\displaystyle ({\hat {\mu }}_{D},{\hat {\mu }}_{G})}$ , which is stable under simultaneous move order: ${\displaystyle {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D})}$


For general games, these equilibria do not have to agree, or even to exist. For the original GAN game, these equilibria all exist, and are all equal. However, for more general GAN games, these do not necessarily exist, or agree.


### Main theorems for GAN game
The original GAN paper proved the following two theorems:

Theorem (the optimal discriminator computes the Jensen–Shannon divergence)—For any fixed generator strategy  ${\displaystyle \mu _{G}}$ , let the optimal reply be  ${\displaystyle D^{*}=\arg \max _{D}L(\mu _{G},D)}$ , then

 ${\displaystyle {\begin{aligned}D^{*}(x)&={\frac {d\mu _{\text{ref}}}{d(\mu _{\text{ref}}+\mu _{G})}}\\[6pt]L(\mu _{G},D^{*})&=2D_{JS}(\mu _{\text{ref}};\mu _{G})-2\ln 2\end{aligned}}}$ 

where the derivative is the Radon–Nikodym derivative, and  ${\displaystyle D_{JS}}$  is the Jensen–Shannon divergence.

By Jensen's inequality,

 ${\displaystyle \operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]\leq \operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln \operatorname {E} _{y\sim \mu _{D}(x)}[y]]}$ 
and similarly for the other term. Therefore, the optimal reply can be deterministic, i.e.  ${\displaystyle \mu _{D}(x)=\delta _{D(x)}}$  for some function  ${\displaystyle D:\Omega \to [0,1]}$ , in which case

 ${\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))].}$ 

To define suitable density functions, we define a base measure  ${\displaystyle \mu :=\mu _{\text{ref}}+\mu _{G}}$ , which allows us to take the Radon–Nikodym derivatives

 ${\displaystyle \rho _{\text{ref}}={\frac {d\mu _{\text{ref}}}{d\mu }}\quad \rho _{G}={\frac {d\mu _{G}}{d\mu }}}$ 
with  ${\displaystyle \rho _{\text{ref}}+\rho _{G}=1}$ .

We then have

 ${\displaystyle L(\mu _{G},\mu _{D}):=\int \mu (dx)\left[\rho _{\text{ref}}(x)\ln(D(x))+\rho _{G}(x)\ln(1-D(x))\right].}$ 

The integrand is just the negative cross-entropy between two Bernoulli random variables with parameters  ${\displaystyle \rho _{\text{ref}}(x)}$  and  ${\displaystyle D(x)}$ . We can write this as  ${\displaystyle -H(\rho _{\text{ref}}(x))-D_{KL}(\rho _{\text{ref}}(x)\parallel D(x))}$ , where  ${\displaystyle H}$  is the binary entropy function, so

 ${\displaystyle L(\mu _{G},\mu _{D})=-\int \mu (dx)(H(\rho _{\text{ref}}(x))+D_{KL}(\rho _{\text{ref}}(x)\parallel D(x))).}$ 

This means that the optimal strategy for the discriminator is  ${\displaystyle D(x)=\rho _{\text{ref}}(x)}$ , with   
 ${\displaystyle L(\mu _{G},\mu _{D}^{*})=-\int \mu (dx)H(\rho _{\text{ref}}(x))=D_{JS}(\mu _{\text{ref}}\parallel \mu _{G})-2\ln 2}$ 

after routine calculation.

Interpretation: For any fixed generator strategy  ${\displaystyle \mu _{G}}$ , the optimal discriminator keeps track of the likelihood ratio between the reference distribution and the generator distribution: ${\displaystyle {\frac {D(x)}{1-D(x)}}={\frac {d\mu _{\text{ref}}}{d\mu _{G}}}(x)={\frac {\mu _{\text{ref}}(dx)}{\mu _{G}(dx)}};\quad D(x)=\sigma (\ln \mu _{\text{ref}}(dx)-\ln \mu _{G}(dx))}$ where  ${\displaystyle \sigma }$  is the logistic function.
In particular, if the prior probability for an image  ${\displaystyle x}$  to come from the reference distribution is equal to  ${\displaystyle {\frac {1}{2}}}$ , then  ${\displaystyle D(x)}$  is just the posterior probability that  ${\displaystyle x}$  came from the reference distribution: ${\displaystyle D(x)=\Pr(x{\text{ came from reference distribution}}\mid x).}$ 

Theorem (the unique equilibrium point)—For any GAN game, there exists a pair  ${\displaystyle ({\hat {\mu }}_{D},{\hat {\mu }}_{G})}$  that is both a sequential equilibrium and a Nash equilibrium:

 ${\displaystyle {\begin{aligned}&L({\hat {\mu }}_{G},{\hat {\mu }}_{D})=\min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=&\max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=-2\ln 2\\[6pt]&{\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D}),&\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})\\[6pt]&{\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),&\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D})\\[6pt]&\forall x\in \Omega ,{\hat {\mu }}_{D}(x)=\delta _{\frac {1}{2}},&\quad {\hat {\mu }}_{G}=\mu _{\text{ref}}\end{aligned}}}$ 

That is, the generator perfectly mimics the reference, and the discriminator outputs  ${\displaystyle {\frac {1}{2}}}$  deterministically on all inputs.

From the previous proposition,

 ${\displaystyle \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=\mu _{\text{ref}};\quad \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=-2\ln 2.}$ 

For any fixed discriminator strategy  ${\displaystyle \mu _{D}}$ , any  ${\displaystyle \mu _{G}}$  concentrated on the set

 ${\displaystyle \{x\mid \operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)]=\inf _{x}\operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)]\}}$ 
is an optimal strategy for the generator. Thus,

 ${\displaystyle \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=\arg \max _{\mu _{D}}\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\inf _{x}\operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)].}$ 

By Jensen's inequality, the discriminator can only improve by adopting the deterministic strategy of always playing  ${\displaystyle D(x)=\operatorname {E} _{y\sim \mu _{D}(x)}[y]}$ . Therefore,

 ${\displaystyle \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=\arg \max _{D}\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\inf _{x}\ln(1-D(x))}$ 

By Jensen's inequality,

 ${\displaystyle {\begin{aligned}&\ln \operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)]+\inf _{x}\ln(1-D(x))\\[6pt]={}&\ln \operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)]+\ln(1-\sup _{x}D(x))\\[6pt]={}&\ln[\operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)](1-\sup _{x}D(x))]\leq \ln[\sup _{x}D(x))(1-\sup _{x}D(x))]\leq \ln {\frac {1}{4}},\end{aligned}}}$ 

with equality if  ${\displaystyle D(x)={\frac {1}{2}}}$ , so

 ${\displaystyle \forall x\in \Omega ,{\hat {\mu }}_{D}(x)=\delta _{\frac {1}{2}};\quad \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=-2\ln 2.}$ 

Finally, to check that this is a Nash equilibrium, note that when  ${\displaystyle \mu _{G}=\mu _{\text{ref}}}$ , we have

 ${\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln(y(1-y))]}$ 
which is always maximized by  ${\displaystyle y={\frac {1}{2}}}$ .

When  ${\displaystyle \forall x\in \Omega ,\mu _{D}(x)=\delta _{\frac {1}{2}}}$ , any strategy is optimal for the generator.


## Training and evaluating GAN

### Training

#### Unstable convergence
While the GAN game has a unique global equilibrium point when both the generator and discriminator have access to their entire strategy sets, the equilibrium is no longer guaranteed when they have a restricted strategy set.

In practice, the generator has access only to measures of form  ${\displaystyle \mu _{Z}\circ G_{\theta }^{-1}}$ , where  ${\displaystyle G_{\theta }}$  is a function computed by a neural network with parameters  ${\displaystyle \theta }$ , and  ${\displaystyle \mu _{Z}}$  is an easily sampled distribution, such as the uniform or normal distribution. Similarly, the discriminator has access only to functions of form  ${\displaystyle D_{\zeta }}$ , a function computed by a neural network with parameters  ${\displaystyle \zeta }$ . These restricted strategy sets take up a vanishingly small proportion of their entire strategy sets.

Further, even if an equilibrium still exists, it can only be found by searching in the high-dimensional space of all possible neural network functions. The standard strategy of using gradient descent to find the equilibrium often does not work for GAN, and often the game "collapses" into one of several failure modes. To improve the convergence stability, some training strategies start with an easier task, such as generating low-resolution images or simple images (one object with uniform background), and gradually increase the difficulty of the task during training. This essentially translates to applying a curriculum learning scheme.


#### Mode collapse
GANs often suffer from mode collapse where they fail to generalize properly, missing entire modes from the input data. For example, a GAN trained on the MNIST dataset containing many samples of each digit might only generate pictures of digit 0. This was termed "the Helvetica scenario".

One way this can happen is if the generator learns too fast compared to the discriminator. If the discriminator  ${\displaystyle D}$  is held constant, then the optimal generator would only output elements of  ${\displaystyle \arg \max _{x}D(x)}$ . So for example, if during GAN training for generating MNIST dataset, for a few epochs, the discriminator somehow prefers the digit 0 slightly more than other digits, the generator may seize the opportunity to generate only digit 0, then be unable to escape the local minimum after the discriminator improves.

Some researchers perceive the root problem to be a weak discriminative network that fails to notice the pattern of omission, while others assign blame to a bad choice of objective function. Many solutions have been proposed, but it is still an open problem.

Even the state-of-the-art architecture, BigGAN (2019), could not avoid mode collapse. The authors resorted to "allowing collapse to occur at the later stages of training, by which time a model is sufficiently trained to achieve good results".


#### Two time-scale update rule
The two time-scale update rule (TTUR) is proposed to make GAN convergence more stable by making the learning rate of the generator lower than that of the discriminator. The authors argued that the generator should move slower than the discriminator, so that it does not "drive the discriminator steadily into new regions without capturing its gathered information".

They proved that a general class of games that included the GAN game, when trained under TTUR, "converges under mild assumptions to a stationary local Nash equilibrium".

They also proposed using the Adam stochastic optimization to avoid mode collapse, as well as the Fréchet inception distance for evaluating GAN performances.


#### Vanishing gradient
Conversely, if the discriminator learns too fast compared to the generator, then the discriminator could almost perfectly distinguish  ${\displaystyle \mu _{G_{\theta }},\mu _{\text{ref}}}$ . In such case, the generator  ${\displaystyle G_{\theta }}$  could be stuck with a very high loss no matter which direction it changes its  ${\displaystyle \theta }$ , meaning that the gradient  ${\displaystyle \nabla _{\theta }L(G_{\theta },D_{\zeta })}$  would be close to zero. In such case, the generator cannot learn, a case of the vanishing gradient problem.

Intuitively speaking, the discriminator is too good, and since the generator cannot take any small step (only small steps are considered in gradient descent) to improve its payoff, it does not even try.

One important method for solving this problem is the Wasserstein GAN.


### Evaluation
GANs are usually evaluated by Inception score (IS), which measures how varied the generator's outputs are (as classified by an image classifier, usually Inception-v3), or Fréchet inception distance (FID), which measures how similar the generator's outputs are to a reference set (as classified by a learned image featurizer, such as Inception-v3 without its final layer). Many papers that propose new GAN architectures for image generation report how their architectures break the state of the art on FID or IS.

Another evaluation method is the Learned Perceptual Image Patch Similarity (LPIPS), which starts with a learned image featurizer  ${\displaystyle f_{\theta }:{\text{Image}}\to \mathbb {R} ^{n}}$ , and finetunes it by supervised learning on a set of  ${\displaystyle (x,x',\operatorname {perceptual~difference} (x,x'))}$ , where  ${\displaystyle x}$  is an image,  ${\displaystyle x'}$  is a perturbed version of it, and  ${\displaystyle \operatorname {perceptual~difference} (x,x')}$  is how much they differ, as reported by human subjects. The model is finetuned so that it can approximate  ${\displaystyle \|f_{\theta }(x)-f_{\theta }(x')\|\approx \operatorname {perceptual~difference} (x,x')}$ . This finetuned model is then used to define  ${\displaystyle \operatorname {LPIPS} (x,x'):=\|f_{\theta }(x)-f_{\theta }(x')\|}$ .

Other evaluation methods are reviewed in.


## Variants
There is a veritable zoo of GAN variants. Some of the most prominent are as follows:


### Conditional GAN
Conditional GANs are similar to standard GANs except they allow the model to conditionally generate samples based on additional information. For example, if we want to generate a cat face given a dog picture, we could use a conditional GAN.

The generator in a GAN game generates  ${\displaystyle \mu _{G}}$ , a probability distribution on the probability space  ${\displaystyle \Omega }$ . This leads to the idea of a conditional GAN, where instead of generating one probability distribution on  ${\displaystyle \Omega }$ , the generator generates a different probability distribution  ${\displaystyle \mu _{G}(c)}$  on  ${\displaystyle \Omega }$ , for each given class label  ${\displaystyle c}$ .

For example, for generating images that look like ImageNet, the generator should be able to generate a picture of cat when given the class label "cat".

In the original paper, the authors noted that GAN can be trivially extended to conditional GAN by providing the labels to both the generator and the discriminator.

Concretely, the conditional GAN game is just the GAN game with class labels provided: ${\displaystyle L(\mu _{G},D):=\operatorname {E} _{c\sim \mu _{C},x\sim \mu _{\text{ref}}(c)}[\ln D(x,c)]+\operatorname {E} _{c\sim \mu _{C},x\sim \mu _{G}(c)}[\ln(1-D(x,c))]}$ where  ${\displaystyle \mu _{C}}$  is a probability distribution over classes,  ${\displaystyle \mu _{\text{ref}}(c)}$  is the probability distribution of real images of class  ${\displaystyle c}$ , and  ${\displaystyle \mu _{G}(c)}$  the probability distribution of images generated by the generator when given class label  ${\displaystyle c}$ .

In 2017, a conditional GAN learned to generate 1000 image classes of ImageNet.


### GANs with alternative architectures
The GAN game is a general framework and can be run with any reasonable parametrization of the generator  ${\displaystyle G}$  and discriminator  ${\displaystyle D}$ . In the original paper, the authors demonstrated it using multilayer perceptron networks and convolutional neural networks. Many alternative architectures have been tried.

Deep convolutional GAN (DCGAN): For both generator and discriminator, uses only deep networks consisting entirely of convolution-deconvolution layers, that is, fully convolutional networks.

Self-attention GAN (SAGAN): Starts with the DCGAN, then adds residually-connected standard self-attention modules to the generator and discriminator.

Variational autoencoder GAN (VAEGAN): Uses a variational autoencoder (VAE) for the generator.

Transformer GAN (TransGAN): Uses the pure transformer architecture for both the generator and discriminator, entirely devoid of convolution-deconvolution layers.

Flow-GAN: Uses flow-based generative model for the generator, allowing efficient computation of the likelihood function.


### GANs with alternative objectives
Many GAN variants are merely obtained by changing the loss functions for the generator and discriminator.

Original GAN:

We recast the original GAN objective into a form more convenient for comparison: ${\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln D(x)]-\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}$ 

Original GAN, non-saturating loss:

This objective for generator was recommended in the original paper for faster convergence. ${\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[\ln D(x)]}$ The effect of using this objective is analyzed in Section 2.2.2 of Arjovsky et al.

Original GAN, maximum likelihood:

 ${\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[({\exp }\circ \sigma ^{-1}\circ D)(x)]}$ where  ${\displaystyle \sigma }$  is the logistic function. When the discriminator is optimal, the generator gradient is the same as in maximum likelihood estimation, even though GAN cannot perform maximum likelihood estimation itself.

Hinge loss GAN: ${\displaystyle L_{D}=-\operatorname {E} _{x\sim p_{\text{ref}}}\left[\min \left(0,-1+D(x)\right)\right]-\operatorname {E} _{x\sim \mu _{G}}\left[\min \left(0,-1-D\left(x\right)\right)\right]}$  ${\displaystyle L_{G}=-\operatorname {E} _{x\sim \mu _{G}}[D(x)]}$ Least squares GAN: ${\displaystyle L_{D}=\operatorname {E} _{x\sim \mu _{\text{ref}}}[(D(x)-b)^{2}]+\operatorname {E} _{x\sim \mu _{G}}[(D(x)-a)^{2}]}$  ${\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[(D(x)-c)^{2}]}$ where  ${\displaystyle a,b,c}$  are parameters to be chosen. The authors recommended  ${\displaystyle a=-1,b=1,c=0}$ .


### Wasserstein GAN (WGAN)
The Wasserstein GAN modifies the GAN game at two points:

- The discriminator's strategy set is the set of measurable functions of type  ${\displaystyle D:\Omega \to \mathbb {R} }$  with bounded Lipschitz norm:  ${\displaystyle \|D\|_{L}\leq K}$ , where  ${\displaystyle K}$  is a fixed positive constant.
- The objective is ${\displaystyle L_{WGAN}(\mu _{G},D):=\operatorname {E} _{x\sim \mu _{G}}[D(x)]-\mathbb {E} _{x\sim \mu _{\text{ref}}}[D(x)]}$


One of its purposes is to solve the problem of mode collapse (see above). The authors claim "In no experiment did we see evidence of mode collapse for the WGAN algorithm".


### GANs with more than two players

#### Adversarial autoencoder
An adversarial autoencoder (AAE) is more autoencoder than GAN. The idea is to start with a plain autoencoder, but train a discriminator to discriminate the latent vectors from a reference distribution (often the normal distribution).


#### InfoGAN
In conditional GAN, the generator receives both a noise vector  ${\displaystyle z}$  and a label  ${\displaystyle c}$ , and produces an image  ${\displaystyle G(z,c)}$ . The discriminator receives image-label pairs  ${\displaystyle (x,c)}$ , and computes  ${\displaystyle D(x,c)}$ .

When the training dataset is unlabeled, conditional GAN does not work directly.

The idea of InfoGAN is to decree that every latent vector in the latent space can be decomposed as  ${\displaystyle (z,c)}$ : an incompressible noise part  ${\displaystyle z}$ , and an informative label part  ${\displaystyle c}$ , and encourage the generator to comply with the decree, by encouraging it to maximize  ${\displaystyle I(c,G(z,c))}$ , the mutual information between  ${\displaystyle c}$  and  ${\displaystyle G(z,c)}$ , while making no demands on the mutual information  ${\displaystyle z}$  between  ${\displaystyle G(z,c)}$ .

Unfortunately,  ${\displaystyle I(c,G(z,c))}$  is intractable in general, The key idea of InfoGAN is Variational Mutual Information Maximization: indirectly maximize it by maximizing a lower bound ${\displaystyle {\hat {I}}(G,Q)=\mathbb {E} _{z\sim \mu _{Z},c\sim \mu _{C}}[\ln Q(c\mid G(z,c))];\quad I(c,G(z,c))\geq \sup _{Q}{\hat {I}}(G,Q)}$ where  ${\displaystyle Q}$  ranges over all Markov kernels of type  ${\displaystyle Q:\Omega _{Y}\to {\mathcal {P}}(\Omega _{C})}$ .



The InfoGAN game is defined as follows:

Three probability spaces define an InfoGAN game:
-  ${\displaystyle (\Omega _{X},\mu _{\text{ref}})}$ , the space of reference images.
-  ${\displaystyle (\Omega _{Z},\mu _{Z})}$ , the fixed random noise generator.
-  ${\displaystyle (\Omega _{C},\mu _{C})}$ , the fixed random information generator.

There are 3 players in 2 teams: generator, Q, and discriminator. The generator and Q are on one team, and the discriminator on the other team.
The objective function is ${\displaystyle L(G,Q,D)=L_{GAN}(G,D)-\lambda {\hat {I}}(G,Q)}$ where  ${\displaystyle L_{GAN}(G,D)=\operatorname {E} _{x\sim \mu _{\text{ref}},}[\ln D(x)]+\operatorname {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z,c)))]}$  is the original GAN game objective, and  ${\displaystyle {\hat {I}}(G,Q)=\mathbb {E} _{z\sim \mu _{Z},c\sim \mu _{C}}[\ln Q(c\mid G(z,c))]}$ 

Generator-Q team aims to minimize the objective, and discriminator aims to maximize it: ${\displaystyle \min _{G,Q}\max _{D}L(G,Q,D)}$ 


#### Bidirectional GAN (BiGAN)
The standard GAN generator is a function of type  ${\displaystyle G:\Omega _{Z}\to \Omega _{X}}$ , that is, it is a mapping from a latent space  ${\displaystyle \Omega _{Z}}$  to the image space  ${\displaystyle \Omega _{X}}$ . This can be understood as a "decoding" process, whereby every latent vector  ${\displaystyle z\in \Omega _{Z}}$  is a code for an image  ${\displaystyle x\in \Omega _{X}}$ , and the generator performs the decoding. This naturally leads to the idea of training another network that performs "encoding", creating an autoencoder out of the encoder-generator pair.

Already in the original paper, the authors noted that "Learned approximate inference can be performed by training an auxiliary network to predict  ${\displaystyle z}$  given  ${\displaystyle x}$ ". The bidirectional GAN architecture performs exactly this.



The BiGAN is defined as follows: 

Two probability spaces define a BiGAN game:
-  ${\displaystyle (\Omega _{X},\mu _{X})}$ , the space of reference images.
-  ${\displaystyle (\Omega _{Z},\mu _{Z})}$ , the latent space.

There are 3 players in 2 teams: generator, encoder, and discriminator. The generator and encoder are on one team, and the discriminator on the other team.
The generator's strategies are functions  ${\displaystyle G:\Omega _{Z}\to \Omega _{X}}$ , and the encoder's strategies are functions  ${\displaystyle E:\Omega _{X}\to \Omega _{Z}}$ . The discriminator's strategies are functions  ${\displaystyle D:\Omega _{X}\to [0,1]}$ .
The objective function is ${\displaystyle L(G,E,D)=\mathbb {E} _{x\sim \mu _{X}}[\ln D(x,E(x))]+\mathbb {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z),z))]}$ 

Generator-encoder team aims to minimize the objective, and discriminator aims to maximize it: ${\displaystyle \min _{G,E}\max _{D}L(G,E,D)}$  

In the paper, they gave a more abstract definition of the objective as: ${\displaystyle L(G,E,D)=\mathbb {E} _{(x,z)\sim \mu _{E,X}}[\ln D(x,z)]+\mathbb {E} _{(x,z)\sim \mu _{G,Z}}[\ln(1-D(x,z))]}$ where  ${\displaystyle \mu _{E,X}(dx,dz)=\mu _{X}(dx)\cdot \delta _{E(x)}(dz)}$  is the probability distribution on  ${\displaystyle \Omega _{X}\times \Omega _{Z}}$  obtained by pushing  ${\displaystyle \mu _{X}}$  forward via  ${\displaystyle x\mapsto (x,E(x))}$ , and  ${\displaystyle \mu _{G,Z}(dx,dz)=\delta _{G(z)}(dx)\cdot \mu _{Z}(dz)}$  is the probability distribution on  ${\displaystyle \Omega _{X}\times \Omega _{Z}}$  obtained by pushing  ${\displaystyle \mu _{Z}}$  forward via  ${\displaystyle z\mapsto (G(x),z)}$ .

Applications of bidirectional models include semi-supervised learning, interpretable machine learning, and neural machine translation.


#### CycleGAN
CycleGAN is an architecture for performing translations between two domains, such as between photos of horses and photos of zebras, or photos of night cities and photos of day cities.



The CycleGAN game is defined as follows:

There are two probability spaces  ${\displaystyle (\Omega _{X},\mu _{X}),(\Omega _{Y},\mu _{Y})}$ , corresponding to the two domains needed for translations fore-and-back.
There are 4 players in 2 teams: generators  ${\displaystyle G_{X}:\Omega _{X}\to \Omega _{Y},G_{Y}:\Omega _{Y}\to \Omega _{X}}$ , and discriminators  ${\displaystyle D_{X}:\Omega _{X}\to [0,1],D_{Y}:\Omega _{Y}\to [0,1]}$ .
The objective function is ${\displaystyle L(G_{X},G_{Y},D_{X},D_{Y})=L_{GAN}(G_{X},D_{X})+L_{GAN}(G_{Y},D_{Y})+\lambda L_{cycle}(G_{X},G_{Y})}$ 

where  ${\displaystyle \lambda }$  is a positive adjustable parameter,  ${\displaystyle L_{GAN}}$  is the GAN game objective, and  ${\displaystyle L_{cycle}}$  is the cycle consistency loss: ${\displaystyle L_{cycle}(G_{X},G_{Y})=E_{x\sim \mu _{X}}\|G_{X}(G_{Y}(x))-x\|+E_{y\sim \mu _{Y}}\|G_{Y}(G_{X}(y))-y\|}$ The generators aim to minimize the objective, and the discriminators aim to maximize it: ${\displaystyle \min _{G_{X},G_{Y}}\max _{D_{X},D_{Y}}L(G_{X},G_{Y},D_{X},D_{Y})}$  

Unlike previous work like pix2pix, which requires paired training data, cycleGAN requires no paired data. For example, to train a pix2pix model to turn a summer scenery photo to winter scenery photo and back, the dataset must contain pairs of the same place in summer and winter, shot at the same angle; cycleGAN would only need a set of summer scenery photos, and an unrelated set of winter scenery photos.


### GANs with particularly large or small scales

#### BigGAN
The BigGAN is essentially a self-attention GAN trained on a large scale (up to 80 million parameters) to generate large images of ImageNet (up to 512 x 512 resolution), with numerous engineering tricks to make it converge.


#### Invertible data augmentation
When there is insufficient training data, the reference distribution  ${\displaystyle \mu _{\text{ref}}}$  cannot be well-approximated by the empirical distribution given by the training dataset. In such cases, data augmentation can be applied, to allow training GAN on smaller datasets. Naïve data augmentation, however, brings its problems.

Consider the original GAN game, slightly reformulated as follows: ${\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}$ Now we use data augmentation by randomly sampling semantic-preserving transforms  ${\displaystyle T:\Omega \to \Omega }$  and applying them to the dataset, to obtain the reformulated GAN game: ${\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}[\ln D(T(x))]-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}$ This is equivalent to a GAN game with a different distribution  ${\displaystyle \mu _{\text{ref}}'}$ , sampled by  ${\displaystyle T(x)}$ , with  ${\displaystyle x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}$ . For example, if  ${\displaystyle \mu _{\text{ref}}}$  is the distribution of images in ImageNet, and  ${\displaystyle \mu _{\text{trans}}}$  samples identity-transform with probability 0.5, and horizontal-reflection with probability 0.5, then  ${\displaystyle \mu _{\text{ref}}'}$  is the distribution of images in ImageNet and horizontally-reflected ImageNet, combined.

The result of such training would be a generator that mimics  ${\displaystyle \mu _{\text{ref}}'}$ . For example, it would generate images that look like they are randomly cropped, if the data augmentation uses random cropping.

The solution is to apply data augmentation to both generated and real images: ${\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}[\ln D(T(x))]-\operatorname {E} _{x\sim \mu _{G},T\sim \mu _{\text{trans}}}[\ln(1-D(T(x)))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G},T\sim \mu _{\text{trans}}}[\ln(1-D(T(x)))]\end{cases}}}$ The authors demonstrated high-quality generation using just 100-picture-large datasets.

The StyleGAN-2-ADA paper points out a further point on data augmentation: it must be invertible. Continue with the example of generating ImageNet pictures. If the data augmentation is "randomly rotate the picture by 0, 90, 180, 270 degrees with equal probability", then there is no way for the generator to know which is the true orientation: Consider two generators  ${\displaystyle G,G'}$ , such that for any latent  ${\displaystyle z}$ , the generated image  ${\displaystyle G(z)}$  is a 90-degree rotation of  ${\displaystyle G'(z)}$ . They would have exactly the same expected loss, and so neither is preferred over the other.

The solution is to only use invertible data augmentation: instead of "randomly rotate the picture by 0, 90, 180, 270 degrees with equal probability", use "randomly rotate the picture by 90, 180, 270 degrees with 0.1 probability, and keep the picture as it is with 0.7 probability". This way, the generator is still rewarded  to keep images oriented the same way as un-augmented ImageNet pictures.

Abstractly, the effect of randomly sampling transformations  ${\displaystyle T:\Omega \to \Omega }$  from the distribution  ${\displaystyle \mu _{\text{trans}}}$  is to define a Markov kernel  ${\displaystyle K_{\text{trans}}:\Omega \to {\mathcal {P}}(\Omega )}$ . Then, the data-augmented GAN game pushes the generator to find some  ${\displaystyle {\hat {\mu }}_{G}\in {\mathcal {P}}(\Omega )}$ , such that  ${\displaystyle K_{\text{trans}}*\mu _{\text{ref}}=K_{\text{trans}}*{\hat {\mu }}_{G}}$ where  ${\displaystyle *}$  is the Markov kernel convolution.
A data-augmentation method is defined to be invertible if its Markov kernel  ${\displaystyle K_{\text{trans}}}$  satisfies ${\displaystyle K_{\text{trans}}*\mu =K_{\text{trans}}*\mu '\implies \mu =\mu '\quad \forall \mu ,\mu '\in {\mathcal {P}}(\Omega )}$ Immediately by definition, we see that composing multiple invertible data-augmentation methods results in yet another invertible method. Also by definition, if the data-augmentation method is invertible, then using it in a GAN game does not change the optimal strategy  ${\displaystyle {\hat {\mu }}_{G}}$  for the generator, which is still  ${\displaystyle \mu _{\text{ref}}}$ .

There are two prototypical examples of invertible Markov kernels:

Discrete case: Invertible stochastic matrices, when  ${\displaystyle \Omega }$  is finite.

For example, if  ${\displaystyle \Omega =\{\uparrow ,\downarrow ,\leftarrow ,\rightarrow \}}$  is the set of four images of an arrow, pointing in 4 directions, and the data augmentation is "randomly rotate the picture by 90, 180, 270 degrees with probability  ${\displaystyle p}$ , and keep the picture as it is with probability  ${\displaystyle (1-3p)}$ ", then the Markov kernel  ${\displaystyle K_{\text{trans}}}$  can be represented as a stochastic matrix: ${\displaystyle [K_{\text{trans}}]={\begin{bmatrix}(1-3p)&p&p&p\\p&(1-3p)&p&p\\p&p&(1-3p)&p\\p&p&p&(1-3p)\end{bmatrix}}}$  and  ${\displaystyle K_{\text{trans}}}$  is an invertible kernel iff  ${\displaystyle [K_{\text{trans}}]}$  is an invertible matrix, that is,  ${\displaystyle p\neq 1/4}$ .

Continuous case: The gaussian kernel, when  ${\displaystyle \Omega =\mathbb {R} ^{n}}$  for some  ${\displaystyle n\geq 1}$ .

For example, if  ${\displaystyle \Omega =\mathbb {R} ^{256^{2}}}$  is the space of 256x256 images, and the data-augmentation method is "generate a gaussian noise  ${\displaystyle z\sim {\mathcal {N}}(0,I_{256^{2}})}$ , then add  ${\displaystyle \epsilon z}$  to the image", then  ${\displaystyle K_{\text{trans}}}$  is just convolution by the density function of  ${\displaystyle {\mathcal {N}}(0,\epsilon ^{2}I_{256^{2}})}$ . This is invertible, because convolution by a gaussian is just convolution by the heat kernel, so given any  ${\displaystyle \mu \in {\mathcal {P}}(\mathbb {R} ^{n})}$ , the convolved distribution  ${\displaystyle K_{\text{trans}}*\mu }$  can be obtained by heating up  ${\displaystyle \mathbb {R} ^{n}}$  precisely according to  ${\displaystyle \mu }$ , then wait for time  ${\displaystyle \epsilon ^{2}/4}$ . With that, we can recover  ${\displaystyle \mu }$  by running the heat equation backwards in time for  ${\displaystyle \epsilon ^{2}/4}$ .

More examples of invertible data augmentations are found in the paper.


#### SinGAN
SinGAN pushes data augmentation to the limit, by using only a single image as training data and performing data augmentation on it. The GAN architecture is adapted to this training method by using a multi-scale pipeline.

The generator  ${\displaystyle G}$  is decomposed into a pyramid of generators  ${\displaystyle G=G_{1}\circ G_{2}\circ \cdots \circ G_{N}}$ , with the lowest one generating the image  ${\displaystyle G_{N}(z_{N})}$  at the lowest resolution, then the generated image is scaled up to  ${\displaystyle r(G_{N}(z_{N}))}$ , and fed to the next level to generate an image  ${\displaystyle G_{N-1}(z_{N-1}+r(G_{N}(z_{N})))}$  at a higher resolution, and so on. The discriminator is decomposed into a pyramid as well.


### StyleGAN series
The StyleGAN family is a series of architectures published by Nvidia's research division.


#### Progressive GAN
Progressive GAN is a method for training GAN for large-scale image generation stably, by growing a GAN generator from small to large scale in a pyramidal fashion. Like SinGAN, it decomposes the generator as ${\displaystyle G=G_{1}\circ G_{2}\circ \cdots \circ G_{N}}$ , and the discriminator as  ${\displaystyle D=D_{1}\circ D_{2}\circ \cdots \circ D_{N}}$ .

During training, at first only  ${\displaystyle G_{N},D_{N}}$  are used in a GAN game to generate 4x4 images. Then  ${\displaystyle G_{N-1},D_{N-1}}$  are added to reach the second stage of GAN game, to generate 8x8 images, and so on, until we reach a GAN game to generate 1024x1024 images.

To avoid shock between stages of the GAN game, each new layer is "blended in" (Figure 2 of the paper). For example, this is how the second stage GAN game starts:

- Just before, the GAN game consists of the pair  ${\displaystyle G_{N},D_{N}}$  generating and discriminating 4x4 images.
- Just after, the GAN game consists of the pair  ${\displaystyle ((1-\alpha )+\alpha \cdot G_{N-1})\circ u\circ G_{N},D_{N}\circ d\circ ((1-\alpha )+\alpha \cdot D_{N-1})}$  generating and discriminating 8x8 images. Here, the functions  ${\displaystyle u,d}$  are image up- and down-sampling functions, and  ${\displaystyle \alpha }$  is a blend-in factor (much like an alpha in image composing) that smoothly glides from 0 to 1.



#### StyleGAN-1
StyleGAN-1 is designed as a combination of Progressive GAN with neural style transfer.

The key architectural choice of StyleGAN-1 is a progressive growth mechanism, similar to Progressive GAN. Each generated image starts as a constant  ${\displaystyle 4\times 4\times 512}$  array, and repeatedly passed through style blocks. Each style block applies a "style latent vector" via affine transform ("adaptive instance normalization"), similar to how neural style transfer uses Gramian matrix. It then adds noise, and normalize (subtract the mean, then divide by the variance).

At training time, usually only one style latent vector is used per image generated, but sometimes two ("mixing regularization") in order to encourage each style block to independently perform its stylization without expecting help from other style blocks (since they might receive an entirely different style latent vector).

After training, multiple style latent vectors can be fed into each style block. Those fed to the lower layers control the large-scale styles, and those fed to the higher layers control the fine-detail styles.

Style-mixing between two images  ${\displaystyle x,x'}$  can be performed as well. First, run a gradient descent to find  ${\displaystyle z,z'}$  such that  ${\displaystyle G(z)\approx x,G(z')\approx x'}$ . This is called "projecting an image back to style latent space". Then,  ${\displaystyle z}$  can be fed to the lower style blocks, and  ${\displaystyle z'}$  to the higher style blocks, to generate a composite image that has the large-scale style of  ${\displaystyle x}$ , and the fine-detail style of  ${\displaystyle x'}$ . Multiple images can also be composed this way.


#### StyleGAN-2
StyleGAN-2 improves upon StyleGAN-1, by using the style latent vector to transform the convolution layer's weights instead, thus solving the "blob" problem.

This was updated by the StyleGAN-2-ADA ("ADA" stands for "adaptive"), which uses invertible data augmentation as described above. It also tunes the amount of data augmentation applied by starting at zero, and gradually increasing it until an "overfitting heuristic" reaches a target level, thus the name "adaptive".


#### StyleGAN-3
StyleGAN-3 improves upon StyleGAN-2 by solving the "texture sticking" problem, which can be seen in the official videos. They analyzed the problem by the Nyquist–Shannon sampling theorem, and argued that the layers in the generator learned to exploit the high-frequency signal in the pixels they operate upon.

To solve this, they proposed imposing strict lowpass filters between each generator's layers, so that the generator is forced to operate on the pixels in a way faithful to the continuous signals they represent, rather than operate on them as merely discrete signals. They further imposed rotational and translational invariance by using more signal filters. The resulting StyleGAN-3 is able to solve the texture sticking problem, as well as generating images that rotate and translate smoothly.


## Other uses
Other than for generative and discriminative modelling of data, GANs have been used for other things.

GANs have been used for transfer learning to enforce the alignment of the latent feature space, such as in deep reinforcement learning. This works by feeding the embeddings of the source and target task to the discriminator which tries to guess the context. The resulting loss is then (inversely) backpropagated through the encoder.


## Applications

### Science
- Iteratively reconstruct astronomical images[55]
- Simulate gravitational lensing for dark matter research.[56][57][58]
- Model the distribution of dark matter in a particular direction in space and to predict the gravitational lensing that will occur.[59][60]
- Model high energy jet formation[61] and showers through calorimeters of high-energy physics experiments.[62][63][64][65]
- Approximate bottlenecks in computationally expensive simulations of particle physics experiments. Applications in the context of present and proposed CERN experiments have demonstrated the potential of these methods for accelerating simulation and/or improving simulation fidelity.[66][67]
- Reconstruct velocity and scalar fields in turbulent flows.[68][69][70]


GAN-generated molecules were validated experimentally in mice.


### Medical
One of the major concerns in medical imaging is preserving patient privacy. Due to these reasons, researchers often face difficulties in obtaining medical images for their research purposes. GAN has been used for generating synthetic medical images, such as MRI and PET images to address this challenge.

GAN can be used to detect glaucomatous images helping the early diagnosis which is essential to avoid partial or total loss of vision.

GANs have been used to create forensic facial reconstructions of deceased historical figures.


### Malicious
Concerns have been raised about the potential use of GAN-based human image synthesis for sinister purposes, e.g., to produce fake, possibly incriminating, photographs and videos.
GANs can be used to generate unique, realistic profile photos of people who do not exist, in order to automate creation of fake social media profiles.

In 2019 the state of California considered and passed on October 3, 2019, the bill AB-602, which bans the use of human image synthesis technologies to make fake pornography without the consent of the people depicted, and bill AB-730, which prohibits distribution of manipulated videos of a political candidate within 60 days of an election. Both bills were authored by Assembly member Marc Berman and signed by Governor Gavin Newsom. The laws went into effect in 2020.

DARPA's Media Forensics program studies ways to counteract fake media, including fake media produced using GANs.


### Fashion, art and advertising
GANs can be used to generate art; The Verge wrote in March 2019 that "The images created by GANs have become the defining look of contemporary AI art." GANs can also be used to

- inpaint photographs[82]
- generate fashion models,[83] shadows,[84] photorealistic renders of interior design, industrial design, shoes, etc.[85] Such networks were reported to be used by Facebook.[86]


Some have worked with using GAN for artistic creativity, as "creative adversarial network". A GAN, trained on a set of 15,000 portraits from WikiArt from the 14th to the 19th century, created the 2018 painting Edmond de Belamy, which sold for US$432,500.

GANs were used by the video game modding community to up-scale low-resolution 2D textures in old video games by recreating them in 4k or higher resolutions via image training, and then down-sampling them to fit the game's native resolution (resembling supersampling anti-aliasing).

In 2020, Artbreeder was used to create the main antagonist in the sequel to the psychological web horror series Ben Drowned. The author would later go on to praise GAN applications for their ability to help generate assets for independent artists who are short on budget and manpower.

In May 2020, Nvidia researchers taught an AI system (termed "GameGAN") to recreate the game of Pac-Man simply by watching it being played.

In August 2019, a large dataset consisting of 12,197 MIDI songs each with paired lyrics and melody alignment was created for neural melody generation from lyrics using conditional GAN-LSTM (refer to sources at GitHub AI Melody Generation from Lyrics).


### Miscellaneous
GANs have been used to 

- show how an individual's appearance might change with age.[96]
- reconstruct 3D models of objects from images,[97]
- generate novel objects as 3D point clouds,[98]
- model patterns of motion in video.[99]
- inpaint missing features in maps, transfer map styles in cartography[100] or augment street view imagery.[101]
- use feedback to generate images and replace image search systems.[102]
- visualize the effect that climate change will have on specific houses.[103]
- reconstruct an image of a person's face after listening to their voice.[104]
- produces videos of a person speaking, given only a single photo of that person.[105]
- recurrent sequence generation.[106]



## History
In 1991, Juergen Schmidhuber published "artificial curiosity", neural networks in a zero-sum game. The first network is a generative model that models a probability distribution over output patterns. The second network learns by gradient descent to predict the reactions of the environment to these patterns. GANs can be regarded as a case where the environmental reaction is 1 or 0 depending on whether the first network's output is in a given set.

Other people had similar ideas but did not develop them similarly. An idea involving adversarial networks was published in a 2010 blog post by Olli Niemitalo. This idea was never implemented and did not involve stochasticity in the generator and thus was not a generative model. It is now known as a conditional GAN or cGAN. An idea similar to GANs was used to model animal behavior by Wei Li, Melvin Gauci and Roderich Gross in 2013.

Another inspiration for GANs was noise-contrastive estimation, which uses the same loss function as GANs and which Goodfellow studied during his PhD in 2010–2014.

Adversarial machine learning has other uses besides generative modeling and can be applied to models other than neural networks. In control theory, adversarial learning based on neural networks was used in 2006 to train robust controllers in a game theoretic sense, by alternating the iterations between a minimizer policy, the controller, and a maximizer policy, the disturbance.

In 2017, a GAN was used for image enhancement focusing on realistic textures rather than pixel-accuracy, producing a higher image quality at high magnification. In 2017, the first faces were generated. These were exhibited in February 2018 at the Grand Palais. Faces generated by StyleGAN in 2019 drew comparisons with Deepfakes.