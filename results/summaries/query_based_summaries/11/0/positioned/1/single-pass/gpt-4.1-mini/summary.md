**Generative Adversarial Network (GAN)** is a framework involving a zero-sum game between two players: a generator and a discriminator. The generator aims to produce samples whose distribution approximates the true data distribution (reference distribution), while the discriminator tries to distinguish real samples from generated ones. Mathematically, the discriminator outputs a function \(D: \Omega \to [0,1]\) assigning probabilities of samples being real, and the generator defines a distribution \(\mu_G\) over \(\Omega\). The objective function is 
\[
L(\mu_G,D) = \mathbb{E}_{x \sim \mu_{\text{ref}}} [\ln D(x)] + \mathbb{E}_{x \sim \mu_G} [\ln(1 - D(x))].
\]
At equilibrium, the optimal discriminator corresponds to the posterior probability that a sample is from the real distribution, and the reciprocal minimizes the Jensen-Shannon divergence between \(\mu_{\text{ref}}\) and \(\mu_G\). The unique equilibrium point is when \(\mu_G = \mu_{\text{ref}}\) and the discriminator outputs 0.5 everywhere.

Training GANs involves alternating optimization where the generator tries to fool the discriminator and the discriminator improves its classification. Practical challenges include unstable convergence, mode collapse (where the generator outputs limited modes), and vanishing gradients (if discriminator becomes too strong). Various techniques such as two time-scale update rules (TTUR) help stabilize training.

**Wasserstein GAN (WGAN)** modifies the GAN formulation by replacing the Jensen-Shannon divergence with the Wasserstein (Earth Mover’s) distance \(W_1\) to provide meaningful gradients even when supports of distributions do not overlap. The discriminator (often called "critic") is constrained to be \(K\)-Lipschitz continuous, turning the objective into 
\[
L_{\text{WGAN}}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)].
\]
By Kantorovich-Rubinstein duality, maximizing over Lipschitz functions approximates the Wasserstein distance between \(\mu_G\) and \(\mu_{\text{ref}}\). Compared to the original GAN, the WGAN’s discriminator offers better gradient signals and reduced variance, addressing vanishing gradient and mode collapse problems. Bounding the Lipschitz norm of the discriminator is commonly enforced by methods such as weight clipping, spectral normalization, or gradient penalty.

WGAN pushes the generator to minimize the Wasserstein distance, resulting in more stable and meaningful training dynamics. This adaptation underlies many modern GAN variants aiming for robust training and richer generative performance.