**Wasserstein GAN (WGAN)** is a variant of the original Generative Adversarial Network (GAN) framework designed to improve training stability and address problems like mode collapse and vanishing gradients inherent in the standard GAN formulation.

The original GAN game involves a generator producing samples and a discriminator distinguishing generated samples from real data, optimizing a Jensen–Shannon divergence-based objective. However, when distributions lie on low-dimensional manifolds (common in high-dimensional data like images), the Jensen–Shannon divergence often becomes infinite, and gradients vanish, preventing effective generator training.

Wasserstein GAN replaces the Jensen–Shannon divergence with the Wasserstein-1 (Earth Mover) distance, which provides a meaningful and smooth measure of distributional difference, even when supports do not overlap. By leveraging the Kantorovich-Rubenstein duality, WGAN formulates the discriminator (critic) as a Lipschitz-continuous function bounded by a Lipschitz constant, aiming to maximize the difference of expected outputs on generated and real data:

\[
L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]
\]

The generator seeks to minimize this value, effectively pushing the generated distribution closer to the real data distribution in Wasserstein distance.

Key differences from standard GAN include:

- **Discriminator (critic) outputs real values (not probabilities) and is constrained to be Lipschitz continuous.**
- **The objective is the Wasserstein distance estimated via the critic, yielding better gradients for the generator.**

To enforce the Lipschitz constraint in the discriminator, several techniques are used:

- **Weight clipping:** constraining weights to small ranges to limit the spectral norm.
- **Spectral normalization:** explicitly upper-bounding the spectral norm (largest singular value) of weight matrices.
- **Gradient penalty:** adding a regularization term penalizing deviations of the discriminator gradient norm from a target value.

WGAN's improved gradient behavior (nonzero almost everywhere) enables more stable training, less mode collapse, and better convergence properties compared to original GANs. It is particularly effective when modeling complex data distributions that lie on lower-dimensional manifolds where traditional divergences fail to provide informative gradients.

In summary, Wasserstein GAN refines the GAN learning framework by adopting the Wasserstein distance and enforcing Lipschitz continuity on the discriminator to provide meaningful, stable learning signals to the generator, overcoming challenges that hinder standard GAN training.