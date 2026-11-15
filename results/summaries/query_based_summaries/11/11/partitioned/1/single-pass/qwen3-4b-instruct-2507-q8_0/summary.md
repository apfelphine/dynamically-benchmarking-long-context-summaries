**Wasserstein GAN (WGAN)** is a generative adversarial network (GAN) that addresses key instability issues in the original GAN by modifying the discriminator's objective and constraints.

- **Core Idea**: Instead of using the Jensen-Shannon divergence (JSD) as the loss function (which can lead to vanishing gradients and mode collapse), WGAN uses the **Wasserstein distance** (also known as Earth Mover's distance). This distance provides a more continuous and meaningful measure of the difference between two probability distributions, resulting in smoother gradients for the generator.

- **Discriminator Constraints**: The discriminator is restricted to functions with a bounded Lipschitz norm (typically set to 1). This ensures that the gradient of the discriminator with respect to the input is bounded, preventing the generator from receiving a vanishing or exploding gradient signal.

- **Objective Function**: The loss function becomes $ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)] $, where $ \mu_G $ is the generator's output distribution and $ \mu_{\text{ref}} $ is the real data distribution. This formulation avoids the problematic JSD and leads to more stable training.

- **Training Methods**:
  - **Weight Clipping**: All weights of the discriminator are clipped to a fixed range (e.g., $[-1, 1]$) to enforce the Lipschitz constraint, though this can be suboptimal.
  - **Spectral Normalization**: A more effective method that normalizes the spectral radius of the weight matrices of the discriminator, ensuring the Lipschitz norm is bounded without clipping.
  - **Gradient Penalty**: Adds a penalty term to the loss function that penalizes the discriminator's gradients exceeding a certain threshold, encouraging the Lipschitz constraint without explicit weight clipping or normalization.

- **Advantages**: WGAN produces more stable training, avoids mode collapse, and provides a more meaningful gradient signal for the generator, leading to better convergence and improved sample quality compared to standard GANs.