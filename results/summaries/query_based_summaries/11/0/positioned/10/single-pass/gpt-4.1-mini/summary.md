**Generative Adversarial Network (GAN)**

A Generative Adversarial Network (GAN) is a zero-sum game between two players: a generator and a discriminator. The generator aims to produce samples from a probability distribution that approximates the real data distribution, while the discriminator tries to distinguish between real data and generated samples. Formally, the generator’s output distribution \(\mu_G\) tries to match the reference distribution \(\mu_{ref}\), and the discriminator outputs values close to 1 for real data and close to 0 for generated data. The game objective is expressed as  
\[
L(\mu_G, D) = \mathbb{E}_{x \sim \mu_{ref}}[\ln D(x)] + \mathbb{E}_{x \sim \mu_G}[\ln (1 - D(x))],
\]
where the generator minimizes \(L\) and the discriminator maximizes it.

In practice, the generator is implemented as a neural network mapping noise vectors from a latent space to data space, and the discriminator is typically a neural network classifying inputs as real or fake. Training involves alternating gradient-based updates of generator and discriminator to improve sample quality and discrimination.

The GAN framework has been extended with conditional GANs which incorporate class labels, alternative architectures like DCGAN and transformer-based GANs, and different loss functions such as hinge loss or least squares loss. 

**Mathematical Properties and Training Challenges:**  
- The optimal discriminator for a fixed generator computes the Jensen-Shannon divergence between generated and real data distributions, and the unique Nash equilibrium is when the generator replicates the real data perfectly and the discriminator outputs 0.5 everywhere.  
- In practice, limited network capacity and training dynamics lead to unstable convergence, mode collapse (generator producing limited diversity), and vanishing gradients. Techniques such as the two time-scale update rule (TTUR) help stabilize training by slowing generator updates relative to discriminator.  
- Evaluation metrics commonly used include Inception Score and Fréchet Inception Distance.

**Wasserstein GAN (WGAN):**  
WGAN modifies the GAN framework by replacing Jensen-Shannon divergence with the Wasserstein distance, aiming to provide a more meaningful gradient for generator training. The discriminator (called "critic") is constrained to be Lipschitz continuous, leading to the objective  
\[
L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)],
\]
with the generator minimizing and the critic maximizing \(L_{WGAN}\). This formulation mitigates mode collapse and vanishing gradients. Techniques to enforce Lipschitz continuity in the critic include weight clipping, spectral normalization, and gradient penalty.

**Advanced GAN Variants:**  
- **Adversarial Autoencoders** combine autoencoders with GAN discriminators to shape latent representations.  
- **InfoGAN** incorporates mutual information maximization to encourage interpretable latent factors.  
- **Bidirectional GAN (BiGAN)** jointly trains encoder and generator to learn inverse mappings between latent and data spaces, facilitating inference.  
- **CycleGAN** enables unpaired image-to-image translation by introducing cycle consistency loss between domains.  
- **StyleGAN series** progressively grow generators/discriminators and incorporate style transfer mechanisms to improve image generation quality and control.

**Applications and Impact:**  
GANs have been widely applied to image synthesis, medical imaging, scientific simulations, anomaly detection, and style transfer. Despite impressive capabilities, GAN-generated media raise concerns about misuse in generating realistic fake content. GAN research continues with efforts to improve stability, sample diversity, training efficiency, and ethical usage.