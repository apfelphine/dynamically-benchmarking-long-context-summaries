The **Wasserstein GAN (WGAN)** is a generative adversarial network (GAN) that addresses key instability issues in the original GAN by replacing the Jensen-Shannon divergence loss with the **Wasserstein distance** (also known as Earth Mover's distance). This change results in a more stable training process and provides a meaningful gradient signal for the generator, even when the generator distribution is far from the real data distribution.

The WGAN defines a zero-sum game between a generator and a discriminator (called a "critic"), where the objective function is the difference in expected values of the critic's output on generated and real data:  
$ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}}[D(x)] $.

The discriminator is constrained to have a bounded Lipschitz norm (i.e., its gradient is bounded), ensuring the Wasserstein distance is well-defined and differentiable. This is achieved through methods such as:
- **Weight clipping**: Directly clipping the weights of the discriminator to keep the Lipschitz norm within a bound.
- **Spectral normalization**: Regularizing the weight matrices to ensure their spectral radius is bounded.
- **Gradient penalty**: Adding a penalty term to the loss function that penalizes deviations from the desired Lipschitz norm.

A major advantage of WGAN over standard GANs is that it avoids the **vanishing gradient problem** and **mode collapse**. In standard GANs, when the generator distribution is far from the real data distribution, the discriminator can become too good at distinguishing real and fake samples, causing the generator to receive no useful gradient signal. WGAN's gradient-based loss ensures a continuous and meaningful signal, enabling the generator to learn more effectively and produce diverse, high-quality samples.

WGAN has been widely adopted due to its stability and effectiveness in generating realistic images and data across various domains.