## Wasserstein GAN Summary

The Wasserstein GAN (WGAN) modifies the standard Generative Adversarial Network (GAN) game by replacing the Jensen–Shannon divergence used in the original GAN with the **Wasserstein metric**. This metric is highly efficient to compute due to the Kantorovich-Rubenstein duality, which states that for a fixed $K>0$, the 1-Wasserstein distance $W_{1}(\mu, \nu)$ can be expressed as:

$$W_{1}(\mu ,\nu )={\frac {1}{K}}\sup _{\|f\|_{L}\leq K}\mathbb {E} _{x\sim \mu }[f(x)]-\mathbb {E} _{y\sim \nu }[f(y)]$$

In a WGAN game, the discriminator (also called the "critic") must be restricted to be a measurable function $D:\Omega \to \mathbb{R}$ with a bounded Lipschitz norm ($\|D\|_{L}\leq K$). The objective function becomes:

$$L_{WGAN}(\mu _{G},D):=\mathbb {E} _{x\sim \mu _{G}}[D(x)]-\mathbb {E} _{x\sim \mu _{ref}}[D(x)]$$

This formulation provides a better gradient signal for the generator compared to the original GAN, which helps avoid the vanishing gradient problem and the issue of mode collapse.

To enforce the Lipschitz norm constraint during training, several methods are employed for the discriminator, including: **weight clipping** (clipping weights to $[-c, c]$), **spectral normalization** (reassigning weights by dividing by their spectral norm), and **gradient penalty** (adding a penalty term to force the gradient of $D(x)$ toward a constant $a$).