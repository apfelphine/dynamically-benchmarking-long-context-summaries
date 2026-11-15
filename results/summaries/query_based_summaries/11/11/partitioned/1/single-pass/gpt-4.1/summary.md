## Wasserstein GAN (WGAN) Summary

Wasserstein GAN (WGAN) modifies the original GAN framework by replacing the Jensen–Shannon divergence with the Wasserstein-1 (Earth Mover) distance as its measure of "farness" between the generator's probability distribution and the reference data distribution. The WGAN game is defined as a zero-sum game with a generator selecting probability measures and a discriminator (or "critic") selecting real-valued functions with bounded Lipschitz norm. The objective for WGAN is:  
**L<sub>WGAN</sub>(μ<sub>G</sub>, D) = E<sub>x ~ μ<sub>G</sub></sub>[D(x)] – E<sub>x ~ μ<sub>ref</sub></sub>[D(x)]**  
By Kantorovich-Rubenstein duality, maximizing discriminator performance pushes the generator to minimize the Wasserstein distance W<sub>1</sub>(μ<sub>G</sub>, μ<sub>ref</sub>). This yields better learning signals when distributions are disjoint, improving training stability and helping to alleviate common GAN issues like vanishing gradients and mode collapse.

For discriminator training, several methods ensure the Lipschitz constraint required by Wasserstein distance computation:  
- **Weight clipping**: Restricts weights within a fixed interval.
- **Spectral normalization**: Scales weights so the spectral norm is bounded (the largest singular value ≤ 1).
- **Gradient penalty**: Adds a penalty term to the loss for deviations of discriminator gradients from a target norm.  

Compared to original GANs, WGAN typically produces better gradients for generator optimization, more stable convergence, and effectively avoids mode collapse. The WGAN framework is widely recognized for its empirical robustness in generative modeling.