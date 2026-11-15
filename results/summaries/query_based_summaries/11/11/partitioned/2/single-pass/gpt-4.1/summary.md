## Wasserstein GAN (WGAN) — Summary

Wasserstein GAN (WGAN) is a variant of the original GAN framework designed to improve stability and gradient flow during training. Unlike traditional GANs that use the Jensen–Shannon divergence as a measure of distance between the real and generated data distributions, WGAN utilizes the Wasserstein (Earth-Mover) distance. The objective function in WGAN is defined by the Kantorovich-Rubenstein duality, where the discriminator (also called "critic") is a real-valued function restricted to have a bounded Lipschitz norm. The WGAN objective is:
${\displaystyle L_{WGAN}(\mu_{G},D) = \mathbb{E}_{x \sim \mu_{G}}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]}$,
with the generator trying to minimize this and the critic aiming to maximize, effectively pushing the generator to directly minimize the Wasserstein distance between generated and real distributions.

For training, the generator optimization is similar to standard gradient descent methods. However, the discriminator must maintain a bounded Lipschitz norm, for which several techniques are employed:
- **Weight Clipping:** Restricts the weights to a fixed interval, limiting the Lipschitz constant.
- **Spectral Normalization:** Normalizes the spectral radius (largest singular value) of the weight matrices to enforce the Lipschitz constraint.
- **Gradient Penalty:** Adds a regularization term penalizing violations in the gradient norm, encouraging Lipschitz continuity.

WGAN provides more meaningful feedback (gradients) to the generator, especially in scenarios with high-dimensional data where standard GANs can suffer from vanishing gradients and mode collapse. The authors of WGAN report successful mitigation of mode collapse issues and improved training behavior compared to standard GANs.