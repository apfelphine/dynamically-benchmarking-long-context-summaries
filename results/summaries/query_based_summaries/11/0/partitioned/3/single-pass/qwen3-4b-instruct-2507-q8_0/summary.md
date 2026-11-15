A **Generative Adversarial Network (GAN)** is a two-player game where a **generator** learns to produce realistic data samples, and a **discriminator** learns to distinguish between real and generated data. The generator aims to minimize the discriminator's ability to detect fakes, while the discriminator aims to maximize its accuracy. This adversarial training leads to the generator producing data that closely mimics the real data distribution.

Key mathematical properties include the **Jensen–Shannon divergence**, where the optimal discriminator computes the likelihood ratio between real and generated data, and a **unique equilibrium** where the generator perfectly mimics the real data distribution and the discriminator outputs 0.5 for all inputs. However, training GANs is often unstable due to issues like **mode collapse** (the generator only produces limited data variations) and **vanishing gradients** (the generator receives no useful feedback).

To improve stability and performance, various **variants** have been developed. These include:
- **Conditional GANs**, which generate data based on labels (e.g., generating a cat image given the label "cat").
- **Wasserstein GAN (WGAN)**, which uses the Wasserstein distance to provide smoother gradients and better convergence, reducing mode collapse.
- **Invertible data augmentation**, which applies transformations to both real and generated data to improve training with limited datasets.
- **StyleGAN series** (e.g., StyleGAN-2, StyleGAN-3), which use progressive growth and style mixing to generate high-quality, diverse images with fine control over style and structure.

GANs have broad applications in image generation, medical imaging, art, and scientific modeling, though they also raise ethical concerns regarding the creation of fake content.