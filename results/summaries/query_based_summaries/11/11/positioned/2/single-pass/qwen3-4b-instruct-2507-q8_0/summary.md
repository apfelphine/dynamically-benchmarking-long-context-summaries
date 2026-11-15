**Wasserstein GAN** is a variant of the Generative Adversarial Network (GAN) that addresses the instability and mode collapse issues prevalent in the original GAN framework. It modifies the GAN game by replacing the original loss function with one based on the **Wasserstein distance** (also known as Earth Mover's distance), which provides a more stable and meaningful measure of the distance between the generator's distribution and the real data distribution.

Key features and differences from the original GAN:

- **Objective Function**: Instead of using the Jensen-Shannon divergence, the Wasserstein GAN uses the objective $L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)]$, which is directly proportional to the Wasserstein distance between the generator's distribution and the reference distribution.
- **Discriminator Constraints**: The discriminator is restricted to functions with bounded Lipschitz norm (i.e., smooth functions), which ensures that the gradients provided to the generator are well-behaved and do not vanish or explode. This is achieved through methods like weight clipping, spectral normalization, or gradient penalty.
- **Stability**: By using the Wasserstein distance, the Wasserstein GAN provides a more stable training process. The gradient provided to the generator is always non-zero, preventing the vanishing gradient problem that plagues the original GAN.
- **Mode Collapse Mitigation**: The Wasserstein GAN is less prone to mode collapse, as the discriminator's feedback is more informative and continuous, leading to better guidance for the generator.
- **Training Methods**: The discriminator's Lipschitz constraint is enforced using techniques such as:
  - **Weight Clipping**: Clipping the weights of the discriminator to a fixed range.
  - **Spectral Normalization**: Normalizing the spectral norm of the weight matrices to ensure bounded Lipschitz norm.
  - **Gradient Penalty**: Adding a penalty term to the loss function that penalizes large gradients, encouraging the discriminator to stay within the Lipschitz bound.

The Wasserstein GAN has proven to be a significant improvement over the original GAN in terms of stability and performance, making it a widely adopted architecture in generative modeling.