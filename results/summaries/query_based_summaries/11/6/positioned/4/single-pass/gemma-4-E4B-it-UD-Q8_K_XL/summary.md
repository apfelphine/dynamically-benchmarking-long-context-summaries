## Autoencoder Summary

An autoencoder is a system composed of two main parts: an encoder, $E_{\phi}: \mathcal{X} \to \mathcal{Z}$, which maps an input message ($\mathbf{x}$) from the input space ($\mathcal{X}$) to a compressed latent space ($\mathcal{Z}$), and a decoder, $D_{\theta}: \mathcal{Z} \to \mathcal{X}$, which attempts to reconstruct the original message ($\mathbf{x}'$). The objective is to minimize a loss function, $L(\theta, \phi) = \mathbb{E}_{x\sim \mu_{ref}}[d(x, D_{\theta}(E_{\phi}(x)))]$, where $d$ measures the reconstruction quality against a reference distribution $\mu_{ref}$.

Different variations of autoencoders serve specialized purposes:
*   **Undercomplete autoencoders** (where $\dim(\mathcal{Z}) < \dim(\mathcal{X})$) are used for dimensionality reduction or feature learning.
*   **Variational Autoencoders (VAEs)** modify this by modeling the latent space as a mixture of distributions, aiming to maximize the Evidence Lower Bound (ELBO) which balances reconstruction error and divergence from a prior distribution.
*   **Sparse Autoencoders (SAE)** encourage the latent codes to be sparse by adding a sparsity regularization loss.
*   **Denoising Autoencoders (DAE)** improve representation by training on intentionally corrupted inputs to recover the original message.
*   **Contractive Autoencoders (CAE)** add regularization to ensure the encoder is robust to infinitesimal input perturbations.

Beyond these, modern variations include **Adversarial Autoencoders (AAE)**, which use a discriminator to enforce that latent vectors match a reference distribution, and **Bidirectional GANs (BiGAN)**, which couple the autoencoder's encoder with a generator.