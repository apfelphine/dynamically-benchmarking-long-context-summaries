**Wasserstein GAN (WGAN)** is a generative adversarial network (GAN) variant that addresses the instability and mode collapse issues of the original GAN by replacing the Jensen-Shannon divergence-based objective with the Wasserstein distance. 

Key features include:
- **Objective Function**: The loss is defined as $L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]$, where $D$ is a critic (discriminator) with a bounded Lipschitz norm $|D|_L \leq K$.
- **Lipschitz Constraint**: The discriminator's strategy set is restricted to functions with bounded Lipschitz norm, ensuring a continuous and well-behaved gradient.
- **Stable Training**: Unlike standard GANs, where gradients can vanish or explode due to the flat gradient of $\ln(1-D(x))$, WGAN provides a gradient of 1 almost everywhere, leading to more stable and reliable training.
- **Training Methods**: Techniques to enforce the Lipschitz constraint include:
  - **Weight Clipping**: Clipping the weights of the discriminator to a fixed range to bound the Lipschitz norm.
  - **Spectral Normalization**: Computing the spectral radius of weight matrices to ensure $|W_i|_s \leq 1$, thereby bounding the Lipschitz norm.
  - **Gradient Penalty**: Adding a penalty term $(\|\nabla D(x)\|_2 - a)^2$ to the loss to regularize the discriminator's gradient and enforce the Lipschitz constraint.
- **Motivation**: The Wasserstein distance provides a more meaningful and continuous measure of the distance between distributions, leading to better convergence and reduced mode collapse compared to the original GAN.

WGAN has been shown to effectively prevent mode collapse and improve training stability, making it a preferred choice for many generative modeling tasks.