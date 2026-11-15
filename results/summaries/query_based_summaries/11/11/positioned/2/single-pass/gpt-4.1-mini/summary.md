### Wasserstein GAN

Wasserstein GAN (WGAN) reformulates the original GAN game by replacing the Jensen–Shannon divergence with the Wasserstein (Earth Mover’s) distance as the measure of difference between the real data distribution and the generator’s distribution. In WGAN, the discriminator—also called the "critic"—is constrained to be a K-Lipschitz continuous function mapping from the data space to the real numbers, rather than outputting probabilities bounded between 0 and 1. Its objective function becomes the expectation of the critic over generated samples minus that over real samples:  
\[
L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)],
\]
where the generator aims to minimize and the critic to maximize this objective. By Kantorovich-Rubinstein duality, optimizing the critic approximates the Wasserstein distance \( W_1(\mu_G, \mu_{ref}) \).

Compared to the original GAN, WGAN provides smoother, more informative gradients to the generator, mitigating issues such as mode collapse and vanishing gradients. Since the Wasserstein metric is continuous and differentiable almost everywhere even when supports of distributions do not overlap, WGAN yields stable learning signals where GAN’s Jensen–Shannon divergence becomes infinite and gradients vanish.

Training WGAN requires enforcing the Lipschitz constraint on the critic. Strategies include weight clipping, spectral normalization (rescaling weights by their largest singular value), or adding a gradient penalty that softly encourages the gradient norm of the critic to be close to a constant. These methods ensure the critic functions lie within the required Lipschitz space, enabling theoretical guarantees of convergence.

In summary, Wasserstein GAN modifies the GAN framework by changing the discriminator’s function space and objective to approximate the Wasserstein distance, thereby providing a more stable training dynamic and improved feedback for generator optimization.