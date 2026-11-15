### Generative Adversarial Networks (GANs)
GANs are zero-sum games between a generator and a discriminator. The generator aims to create realistic data, while the discriminator learns to distinguish real from fake. The game is mathematically defined with an objective function involving log-likelihoods. GANs are implicit generative models, lacking explicit likelihood functions. They generate data in one pass, unlike autoregressive models. GANs use neural networks with a reparametrization trick to avoid intractable probabilistic computations. Training is unstable due to mode collapse and vanishing gradients. Solutions include the two-time-scale update rule, Wasserstein GAN, and curriculum learning. Evaluation uses metrics like Inception Score and Fréchet Inception Distance.

### Variants and Architectures
Key GAN variants include Conditional GANs (using labels), Wasserstein GAN (using Wasserstein distance), and InfoGAN (maximizing mutual information between latent variables and labels). Other architectures include BiGAN (for encoding and decoding), CycleGAN (for domain translation), and StyleGAN (for high-resolution image generation). These variants improve stability, control, and quality.

### Applications
GANs are used in image generation, scientific modeling (e.g., astronomical image reconstruction), medical imaging (synthetic data), fashion and art (generating images), and malicious applications (deepfakes). They have also been used for anomaly detection and data augmentation.

### Training and Evaluation
Training GANs is challenging due to instability. The generator's task is to match the reference distribution, while the discriminator outputs a probability. Evaluation metrics include Inception Score (IS) and Fréchet Inception Distance (FID), which measure diversity and similarity of generated images. Other metrics like LPIPS assess perceptual differences.

### History and Related Models
GANs were inspired by earlier work on adversarial networks and noise-contrastive estimation. They have evolved alongside other generative models like variational autoencoders (VAEs). GANs have been pivotal in advancing generative modeling, with applications ranging from art to scientific research.