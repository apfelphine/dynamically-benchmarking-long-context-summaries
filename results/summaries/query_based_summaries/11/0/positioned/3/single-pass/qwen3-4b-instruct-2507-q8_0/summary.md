A **Generative Adversarial Network (GAN)** is a type of machine learning model based on a zero-sum game between two neural networks: a generator and a discriminator. The generator creates synthetic data samples, aiming to mimic real data, while the discriminator evaluates whether a sample is real or fake. The generator is trained to fool the discriminator, and the discriminator is trained to better distinguish real from fake data. This adversarial process drives both networks to improve iteratively.

The original GAN formulation uses a loss function involving the log-likelihood of real and generated data. However, training GANs is often unstable due to issues like vanishing gradients, mode collapse (where the generator produces only a limited variety of outputs), and poor convergence. To address these problems, variants have been developed, such as:

- **Wasserstein GAN (WGAN)**: Uses a Wasserstein distance instead of Jensen-Shannon divergence, which provides a more stable gradient and reduces mode collapse.
- **Conditional GANs**: Allow generation of data based on additional input (e.g., class labels).
- **CycleGAN**: Enables translation between two domains without paired training data by using cycle consistency loss.
- **InfoGAN**: Encourages the generator to learn disentangled latent variables by maximizing mutual information between latent variables and generated data.
- **Bidirectional GAN (BiGAN)**: Includes an encoder that learns to reconstruct the input from the generated output, forming a latent space mapping.

GANs have diverse applications, including image generation, style transfer, video generation, medical imaging, and even scientific simulations. They are also used in tasks like image inpainting, super-resolution, and anomaly detection. Despite their power, concerns exist about the misuse of GANs for creating fake media, leading to regulatory efforts to limit their use in generating deepfakes.