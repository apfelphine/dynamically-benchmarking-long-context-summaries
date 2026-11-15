## Generative Adversarial Network (GAN) Summary

A Generative Adversarial Network (GAN) is defined by a zero-sum game involving two players: a generator ($\mu_G$) and a discriminator ($\mu_D$). The generator's goal is to minimize an objective function, aiming for its output distribution ($\mu_G$) to closely match a reference distribution ($\mu_{\text{ref}}$). Conversely, the discriminator aims to maximize the objective by outputting a value close to 1 if an input is from $\mu_{\text{ref}}$, and close to 0 if it is from $\mu_G$. In practice, the generative network creates candidates while the discriminative network evaluates them, creating a contest based on data distributions.

Mathematically, the original GAN game objective is $L(\mu_G,\mu_D)=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\operatorname {E} _{x\sim \mu _{G},y\sim \mu _{D}(x)}[\ln(1-y)]$. The optimal discriminator is shown to compute the Jensen–Shannon divergence. Due to practical constraints (restricted strategy sets), training can be unstable, potentially suffering from mode collapse or vanishing gradients. To improve stability, variants like Wasserstein GAN (WGAN) replace the objective with a metric that provides a better gradient signal.

### Key Variants and Concepts:

*   **Conditional GAN (cGAN):** Allows conditional sample generation by providing additional information (like class labels) to both the generator and discriminator.
*   **WGAN:** Modifies the game by restricting the discriminator's strategy set to functions with a bounded Lipschitz norm, using the Wasserstein metric to provide stable training.
*   **CycleGAN:** Enables domain translation (e.g., horse to zebra) between two domains without requiring paired training data, enforced by a cycle consistency loss.
*   **Invertible Data Augmentation:** Using invertible transformations ($\mu_{\text{trans}}$) in the GAN game does not change the optimal generator strategy, which remains $\mu_{\text{ref}}$.
*   **StyleGAN Series:** Introduces architectural improvements, such as progressive growth (Progressive GAN) and style latent vectors, to achieve stable, large-scale image generation.