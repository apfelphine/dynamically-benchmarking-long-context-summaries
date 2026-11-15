## Autoencoder Summary

An autoencoder is a model composed of an encoder ($E_{\phi}: \mathcal{X} \rightarrow \mathcal{Z}$) and a decoder ($D_{\theta}: \mathcal{Z} \rightarrow \mathcal{X}$). The encoder maps an input message $\mathbf{x}$ from the input space $\mathcal{X}$ to a latent representation (code) $\mathbf{z}$ in the latent space $\mathcal{Z}$, while the decoder reconstructs the message $\mathbf{x}'$ from $\mathbf{z}$. The training objective is to minimize a loss function $L(\theta, \phi) = \mathbb{E}_{x \sim \mu_{\text{ref}}}[d(x, D_{\theta}(E_{\phi}(x)))]$, where $d$ measures the reconstruction error, and $\mu_{\text{ref}}$ is the reference probability distribution.

Autoencoders are used for tasks such as dimensionality reduction (undercomplete autoencoders, where $\text{dim}(\mathcal{Z}) < \text{dim}(\mathcal{X})$), information retrieval (e.g., semantic hashing), and anomaly detection (where high reconstruction error signals an anomaly). Variations include:
*   **Denoising Autoencoder (DAE):** Trained by intentionally corrupting inputs during training to recover the original message.
*   **Sparse Autoencoder (SAE):** Enforces sparsity in the latent codes to improve classification performance.
*   **Variational Autoencoder (VAE):** A generative model that models the data distribution $p_{\theta}(x)$ using a latent space composed of mixtures of distributions, optimized by maximizing the Evidence Lower Bound (ELBO).
*   **Contractive Autoencoder (CAE):** Adds a contractive regularization loss to ensure the extracted features resist infinitesimal input perturbations.
*   **Minimum Description Length Autoencoder (MDL-AE):** Incorporates the MDL principle by minimizing the total description length of the latent representation and the reconstruction error.