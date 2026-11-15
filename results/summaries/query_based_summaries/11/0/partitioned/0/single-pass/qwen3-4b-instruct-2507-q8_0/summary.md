A **Generative Adversarial Network (GAN)** is a zero-sum game between two neural networks: a **generator** and a **discriminator**. The generator aims to produce data that mimics a real dataset, while the discriminator learns to distinguish between real and generated data. The game is defined mathematically by an objective function involving expectations over real and generated data distributions, with the generator minimizing and the discriminator maximizing this function.

Key theoretical results show that under ideal conditions, the game converges to a unique equilibrium where the generator perfectly mimics the real data distribution, and the discriminator outputs 0.5 for all inputs, indicating no ability to distinguish between real and fake data. The optimal discriminator computes the Jensen-Shannon divergence between the real and generated distributions.

In practice, GANs face challenges like **mode collapse** (where the generator produces limited variations) and **vanishing gradients** (when the discriminator becomes too powerful, providing no useful feedback). These are mitigated through techniques such as the **two-time-scale update rule**, **Wasserstein GAN (WGAN)**, and **conditional GANs** that incorporate labels.

Variants of GANs include:
- **WGAN**: Uses the Wasserstein distance and bounded Lipschitz discriminators, improving training stability and reducing mode collapse.
- **Conditional GANs**: Allow generation conditioned on labels (e.g., generating a cat image given a "cat" label).
- **Bidirectional GANs (BiGAN)**: Introduce an encoder to create a latent space, enabling reconstruction and learning of data representations.
- **CycleGAN**: Enables domain translation (e.g., converting horse images to zebra images) without paired data, using cycle consistency loss.
- **InfoGAN**: Encourages the latent space to contain interpretable variables by maximizing mutual information between latent variables and generated outputs.

GANs are widely used in image generation, scientific simulations (e.g., reconstructing astronomical images), medical imaging (synthetic data), and art creation. Their ability to learn complex data distributions makes them a foundational tool in generative modeling.