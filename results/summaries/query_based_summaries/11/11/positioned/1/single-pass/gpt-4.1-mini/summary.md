# Summary of Wasserstein GAN

Wasserstein GAN (WGAN) modifies the original GAN framework by replacing the Jensen–Shannon divergence with the Wasserstein distance as the measure of difference between the generator distribution and the real data distribution. This is formalized using the Kantorovich-Rubenstein duality, which allows the Wasserstein-1 distance (Earth Mover's distance) to be efficiently computed as a supremum over 1-Lipschitz functions serving as discriminators (also called critics). The WGAN game is a zero-sum game where the discriminator is constrained to be K-Lipschitz, and the objective function is the difference of expected discriminator outputs over generated and real samples:  
\[ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)] \]  
with the generator minimizing and discriminator maximizing this objective.

WGAN addresses key issues in standard GAN training, such as mode collapse and vanishing gradients. The Wasserstein distance provides a smoother, more meaningful metric with non-vanishing gradients almost everywhere, unlike the Jensen–Shannon divergence which saturates when real and generated distributions fail to overlap in high-dimensional spaces (a typical case for images). This results in more stable training and better gradient flow for the generator.

Training a WGAN requires enforcing the Lipschitz constraint on the discriminator. Several approaches exist:  
- **Weight Clipping:** Directly clipping discriminator weights within fixed bounds to bound the Lipschitz constant, as proposed in the original WGAN.  
- **Spectral Normalization:** Controlling the spectral norm (largest singular value) of each weight matrix to bound the Lipschitz constant efficiently.  
- **Gradient Penalty:** Adding a penalty term to the discriminator loss that encourages the norm of the discriminator's gradient with respect to its input to be close to a target value, indirectly enforcing Lipschitz continuity.

These methods ensure that the discriminator functions remain within the required Lipschitz space, enabling valid approximation of the Wasserstein distance and stable adversarial training.

In comparison to traditional GANs, WGAN discriminators better capture the distance between distributions with lower variance gradients, leading to improved convergence properties. WGAN has been demonstrated to avoid mode collapse under extensive experimental conditions, representing a significant advance in training generative adversarial models.