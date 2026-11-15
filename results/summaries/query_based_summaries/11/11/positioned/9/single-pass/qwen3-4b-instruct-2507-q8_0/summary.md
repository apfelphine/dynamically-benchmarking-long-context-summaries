A **Wasserstein GAN (WGAN)** is a generative model that improves upon the original GAN by using the **Wasserstein distance** (also known as Earth Mover's distance) as the objective function instead of the Jensen-Shannon divergence. This change addresses key issues in standard GANs, such as unstable training and mode collapse.

In WGAN:
- The **discriminator** is constrained to have a **bounded Lipschitz norm**, ensuring that its gradients are well-behaved and provide a stable signal to the generator.
- The objective function becomes $ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)] $, which is directly proportional to the Wasserstein distance between the generator's distribution and the reference data distribution.
- This formulation provides a **better gradient signal** for the generator, especially in regions where the data distribution is sparse or complex, leading to more stable and effective training.

WGAN achieves better convergence and avoids mode collapse compared to traditional GANs. Training the discriminator involves either:
- **Weight clipping** (clipping the weights to a fixed range),
- **Spectral normalization** (normalizing the spectral radius of weight matrices), or
- **Gradient penalty** (adding a penalty term to enforce the Lipschitz constraint).

These methods ensure the discriminator remains within the Lipschitz bound, making the training process more robust and reliable. WGAN has become a foundational architecture in modern generative modeling.