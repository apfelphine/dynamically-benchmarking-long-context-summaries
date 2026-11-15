The **Wasserstein GAN (WGAN)** modifies the traditional GAN framework by replacing the Jensen-Shannon divergence-based loss with the **Wasserstein distance**, a metric that better captures the geometric structure of probability distributions. Unlike the original GAN, which relies on a non-differentiable divergence measure, WGAN uses a **Lipschitz-constrained discriminator** and an objective function that directly measures the distance between the generator's distribution and the reference distribution:

$$
L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G} [D(x)] - \mathbb{E}_{x \sim \mu_{ref}} [D(x)]
$$

### **Key Advantages:**
1. **Stable Training:** WGAN avoids the vanishing gradient problem associated with the Jensen-Shannon divergence, leading to more stable training compared to traditional GANs.
2. **Better Gradient Information:** The discriminator in WGAN provides a more informative gradient signal, allowing the generator to better approximate the reference distribution.
3. **Avoids Mode Collapse:** WGAN’s use of the Wasserstein distance helps prevent mode collapse, a common issue in GANs where the generator fails to generate diverse samples.

### **Training Methods:**
To ensure the discriminator remains Lipschitz-constrained, WGAN employs several techniques:
- **Weight Clipping:** Limits the weights of the discriminator to a bounded range, ensuring the Lipschitz norm constraint is satisfied.
- **Spectral Normalization:** Normalizes the weights of the discriminator to enforce a bounded Lipschitz norm, improving gradient stability.
- **Gradient Penalty:** Adds a penalty term to the loss function to enforce the Lipschitz constraint, encouraging the discriminator to behave like a smooth function.

### **Comparison with GAN:**
The Wasserstein GAN differs fundamentally from the original GAN in its discriminator’s design and loss function. While the original GAN uses a non-differentiable divergence measure (Jensen-Shannon divergence), WGAN relies on a **Lipschitz-constrained discriminator** and a differentiable Wasserstein distance-based loss. This design choice leads to more stable training and better generalization, particularly in high-dimensional spaces like image generation.