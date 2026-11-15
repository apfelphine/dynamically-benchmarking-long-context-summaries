# Summary of Wasserstein GAN (WGAN)

Wasserstein GAN modifies the original GAN framework by replacing the Jensen–Shannon divergence with the Wasserstein-1 distance (Earth Mover’s distance) as the measure of discrepancy between the real data distribution and the generator distribution. Formally, WGAN is defined over a metric probability space with the discriminator (also called critic) constrained to be K-Lipschitz continuous functions. The discriminator’s objective is to maximize the difference of expected values of its outputs on generated samples versus real samples, while the generator aims to minimize this value. By Kantorovich-Rubinstein duality, the WGAN objective directly approximates the Wasserstein distance, providing a meaningful and smooth gradient for the generator even when supports of the distributions do not overlap.

Compared to original GANs, WGAN's discriminator yields better behaved gradients and mitigates issues like mode collapse and vanishing gradients. While the GAN discriminator’s output saturates and results in flat gradients, the WGAN critic’s gradients remain informative and stable, enhancing training dynamics. This is particularly beneficial in high-dimensional settings where the data distribution lies on a low-dimensional manifold, making the Jensen–Shannon divergence problematic (it becomes infinite and gradients vanish). The generator in WGAN uses gradient descent on the critic's score, which has gradient approximately 1 almost everywhere, unlike GAN where the generator gradient can be zero in large regions.

Training WGANs involves carefully enforcing the Lipschitz constraint on the discriminator. Practical methods include:

- **Weight Clipping:** Directly clipping the discriminator's neural network weights within a fixed range to bound its Lipschitz constant, as proposed in the original WGAN paper.

- **Spectral Normalization:** Normalizing weight matrices by their spectral norm (largest singular value) to precisely bound the Lipschitz constant of each layer, allowing stable training without manual clipping.

- **Gradient Penalty:** Adding a regularization term that penalizes deviations of the gradient norm of the discriminator's output w.r.t. its input from a target value, encouraging the discriminator to satisfy the Lipschitz condition softly.

These constraints ensure the discriminator remains a valid critic approximating the Wasserstein distance while maintaining stable and effective gradient propagation during training.

---

This summary covers the motivation, mathematical definition, comparative advantages, and practical training techniques of the Wasserstein GAN strictly based on the provided text.