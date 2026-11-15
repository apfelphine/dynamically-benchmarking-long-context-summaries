**Wasserstein GAN (WGAN)** is a generative adversarial network (GAN) variant that addresses key instability issues in the original GAN by replacing the Jensen-Shannon divergence-based objective with the **Wasserstein distance** (also known as Earth Mover's distance). This change provides a more stable training process by ensuring the discriminator's gradient is bounded and continuous, avoiding the problematic "vanishing gradient" and "mode collapse" issues common in standard GANs.

Key features and innovations include:
- **Objective Function**: The loss function is defined as $L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]$, where $D$ is a Lipschitz-continuous function.
- **Lipschitz Constraint**: The discriminator's strategy set is restricted to functions with a bounded Lipschitz norm ($|D|_L \leq K$), ensuring the gradient of the loss with respect to the generator's output remains stable and non-zero.
- **Stability**: Unlike the original GAN, where the discriminator's gradient can vanish in regions of high divergence, the Wasserstein GAN's discriminator has a gradient of 1 almost everywhere, leading to a more consistent and reliable learning signal for the generator.
- **Training Methods**: To enforce the Lipschitz constraint, several techniques are used:
  - **Weight Clipping**: Directly limits the weights of the discriminator network.
  - **Spectral Normalization**: Regularizes the weight matrices to ensure their spectral norm is bounded.
  - **Gradient Penalty**: Adds a penalty term to the loss function that penalizes large gradients, encouraging the discriminator to maintain a Lipschitz norm.

WGAN has been shown to be effective in practice, with experiments demonstrating a significant reduction in mode collapse and improved training stability, making it a preferred choice for many modern GAN applications.