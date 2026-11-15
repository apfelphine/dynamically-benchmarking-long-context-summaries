**Wasserstein GAN (WGAN)** is a generative adversarial network (GAN) variant that addresses key instability issues in the original GAN by replacing the Jensen-Shannon divergence loss with the **Wasserstein distance** (also known as Earth Mover's distance). This change provides a more stable and meaningful gradient signal for the generator, as the Wasserstein distance measures the minimal "work" required to transform one probability distribution into another, avoiding the problematic vanishing or exploding gradients seen in standard GANs.

The WGAN game is defined over a metric space with a bounded Lipschitz norm for the discriminator. The objective function becomes the difference in expected values of the discriminator between the generated and real data distributions:  
$ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)] $.  

By the Kantorovich-Rubinstein duality theorem, the optimal discriminator output is directly proportional to the Wasserstein distance, enabling the generator to learn a distribution that minimizes this distance to the real data distribution. This results in a more stable training process and better convergence.

To enforce the bounded Lipschitz norm of the discriminator, several methods are employed:
- **Weight clipping**: Directly constrains the weights of the discriminator network.
- **Spectral normalization**: Regularizes the weight matrices to ensure their spectral radius is bounded, thereby limiting the Lipschitz norm.
- **Gradient penalty**: Adds a penalty term to the loss function based on the gradient norm of the discriminator, encouraging it to stay within the desired Lipschitz bound.

WGAN significantly improves upon the original GAN by providing a more robust and reliable training process, making it particularly effective for generating high-quality images and solving problems where mode collapse or unstable gradients are common.