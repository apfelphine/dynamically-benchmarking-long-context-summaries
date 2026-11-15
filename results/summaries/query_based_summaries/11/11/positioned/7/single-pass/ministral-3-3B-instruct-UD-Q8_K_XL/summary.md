The **Wasserstein GAN (WGAN)** is a generative adversarial network framework that addresses key limitations of traditional GANs, particularly mode collapse and unstable training. Unlike the original GAN, which uses the Jensen-Shannon divergence to measure the difference between the generator's distribution and the reference distribution, the Wasserstein GAN employs the **Wasserstein metric** (also known as Earth Mover's Distance), which provides a more stable and efficient gradient-based optimization approach.

### **Key Features of Wasserstein GAN:**
1. **Wasserstein Distance Objective:**
   - The Wasserstein GAN game is defined by minimizing the Wasserstein distance between the generator’s distribution (𝜇_G) and the reference distribution (𝜇_ref).
   - The objective function is:
     \[
     L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G} [D(x)] - \mathbb{E}_{x \sim \mu_{ref}} [D(x)]
     \]
   - This formulation encourages the generator to produce samples that closely resemble the reference distribution, avoiding mode collapse by focusing on the "distance" in a more robust way.

2. **Lipschitz Constraint:**
   - The discriminator in a Wasserstein GAN is constrained to have a bounded Lipschitz norm (𝑁𝑒𝑟𝑒𝑠 ≤ 𝑁), ensuring that its gradients are bounded. This constraint stabilizes training by preventing the discriminator from becoming too sensitive to small input perturbations.
   - Methods to enforce this constraint include:
     - **Weight Clipping:** Restricts the weights of the discriminator to a bounded range.
     - **Spectral Normalization:** Normalizes the weights of the discriminator to ensure their spectral norm is ≤ 1.
     - **Gradient Penalty:** Penalizes deviations from the Lipschitz constraint during training.

3. **Comparison with Traditional GAN:**
   - Traditional GANs rely on the Jensen-Shannon divergence, which can lead to unstable training and mode collapse, especially when the generator’s distribution is far from the reference distribution.
   - Wasserstein GANs use the Wasserstein metric, which is more stable and avoids the vanishing gradient problem by providing a better gradient signal for the generator.

4. **Advantages:**
   - **Stable Training:** The Wasserstein distance provides a more stable gradient signal, reducing the risk of mode collapse and unstable convergence.
   - **Better Generalization:** The use of the Wasserstein metric allows the generator to better approximate the reference distribution, leading to more realistic and diverse samples.
   - **Efficiency:** The Kantorovich-Rubenstein duality theorem allows efficient computation of the Wasserstein distance, making the training process more computationally feasible.

### **Training Process:**
- The generator and discriminator are trained alternately, with the generator minimizing the Wasserstein distance and the discriminator maximizing it.
- The discriminator is constrained to have a bounded Lipschitz norm, which is enforced using one of the methods mentioned above.
- The training process is more stable and less prone to mode collapse compared to traditional GANs, making it a preferred choice for many generative tasks.