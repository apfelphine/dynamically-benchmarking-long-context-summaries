The **Wasserstein GAN (WGAN)** is a variant of the Generative Adversarial Network (GAN) designed to address instability issues in training, particularly mode collapse and vanishing gradients. It modifies the original GAN game by replacing the Jensen-Shannon divergence-based loss with the **Wasserstein distance**, which provides a more stable and meaningful gradient signal for the generator.

Key features of WGAN:
- **Objective Function**: Uses the Wasserstein distance as the loss function: $L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]$.
- **Discriminator Constraint**: The discriminator is restricted to functions with bounded Lipschitz norm, ensuring the gradient of the loss with respect to the generator's input remains well-behaved and non-zero.
- **Stability**: The Wasserstein distance offers a more continuous and robust measure of distance between distributions, preventing the generator from receiving a flat or zero gradient, which helps avoid mode collapse and improves convergence.
- **Training Methods**: To enforce the Lipschitz constraint, WGAN employs techniques such as:
  - **Weight clipping**: Directly limits the weights of the discriminator to a fixed range.
  - **Spectral normalization**: Regularizes the weight matrices to ensure their spectral norm is bounded.
  - **Gradient penalty**: Adds a penalty term to the loss that encourages the discriminator's gradient to remain close to a fixed norm.

WGAN has proven to be more stable and effective than the original GAN, especially in generating high-quality images and handling complex data distributions. It is widely used in image generation and other generative modeling tasks.