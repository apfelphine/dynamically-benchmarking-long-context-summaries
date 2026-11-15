# Summary of Generative Adversarial Network (GAN)

Generative adversarial networks (GANs) are a class of implicit generative models formulated as a two-player zero-sum game between a generator and a discriminator. The generator aims to produce samples whose distribution approximates a true data distribution, while the discriminator attempts to distinguish between real samples and generated ones. Mathematically, the GAN objective is to minimize over the generator and maximize over the discriminator the function:
  
\[
L(G, D) = \mathbb{E}_{x \sim \mu_{ref}}[\ln D(x)] + \mathbb{E}_{z \sim \mu_Z}[\ln(1 - D(G(z)))]
\]

where \(\mu_{ref}\) is the real data distribution and \(\mu_Z\) is a fixed noise distribution. The discriminator outputs values close to 1 for real samples and close to 0 for generated samples. The unique equilibrium occurs when the generator perfectly replicates the real data distribution and the discriminator outputs 0.5 for all inputs.

GAN training involves alternating updates to generator and discriminator networks, usually implemented as deep neural networks. Despite the conceptual clarity, practical training is challenging due to unstable convergence, mode collapse (generator produces limited diversity), and vanishing gradients (discriminator becomes too good). Techniques such as the two time-scale update rule (TTUR) improve stability.

Many variants exist:

- **Conditional GAN:** Allows conditioning generation on additional inputs like class labels.
- **Wasserstein GAN (WGAN):** Replaces Jensen-Shannon divergence with Wasserstein distance to provide more meaningful gradients, improving training stability and mitigating mode collapse.
- **Adversarial Autoencoder:** Combines autoencoders with adversarial training to regularize latent representations.
- **InfoGAN:** Maximizes mutual information between latent codes and generated samples to learn interpretable representations.
- **Bidirectional GAN (BiGAN):** Includes an encoder mapping data to latent space, enabling learned inference.
- **CycleGAN:** Enables unpaired image-to-image translation by enforcing cycle consistency between two domains.

Practical applications include image generation, medical image synthesis, anomaly detection, drug discovery (GAN-generated molecules validated in mice), art creation, privacy-preserving data synthesis, and more. Despite extensive research, GANs are still sensitive to hyperparameters and data, but remain powerful tools for generating realistic synthetic data.

---

This summary focuses exclusively on GAN-related content from the provided text, outlining its formal definition, training challenges, mathematical properties, main variants, and applications.