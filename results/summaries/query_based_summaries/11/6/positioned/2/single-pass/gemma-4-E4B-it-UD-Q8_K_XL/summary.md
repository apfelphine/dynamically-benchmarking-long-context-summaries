## Autoencoder Summary

An autoencoder is composed of two main parts: an encoder ($E_{\phi}: \mathcal{X}\rightarrow \mathcal{Z}$) and a decoder ($D_{\theta}: \mathcal{Z}\rightarrow \mathcal{X}$). The encoder maps input data ($\mathbf{x} \in \mathcal{X}$) to a low-dimensional code or latent variable ($\mathbf{z}$), while the decoder reconstructs the message ($\mathbf{x}' = D_{\theta}(\mathbf{z})$). Typically, both functions are multilayer perceptrons (MLPs).

The quality of an autoencoder is judged by minimizing a loss function, $L(\theta, \phi) = \mathbb{E}_{x\sim \mu_{ref}}[d(x, D_{\theta}(E_{\phi}(x)))]$, where $d(x, x')$ is the reconstruction quality function relative to a reference distribution $\mu_{ref}$. To prevent the encoder and decoder from simply duplicating the signal, the code space ($\mathcal{Z}$) is usually kept smaller than the message space ($\mathcal{X}$), resulting in an undercomplete autoencoder that compresses the message.

Variations on the basic autoencoder include:
*   **Variational Autoencoder (VAE):** A generative model that uses a probabilistic approach, approximating the intractable true posterior distribution $p_{\theta}(z|x)$ with an approximate posterior $q_{\phi}(z|x)$, optimized using the Evidence Lower Bound (ELBO).
*   **Sparse Autoencoder (SAE):** A variant where the codes $\mathbf{E}_{\phi}(x)$ are encouraged to be sparse, either by clamping activations or by adding a sparsity regularization loss.
*   **Denoising Autoencoder (DAE):** Trained by intentionally corrupting the inputs during training and requiring the autoencoder to recover the original, clean message.
*   **Contractive Autoencoder (CAE):** Adds a contractive regularization loss to the standard loss to ensure that small variations in the input lead to small variations in the code.
*   **Minimum Description Length Autoencoder (MDL-AE):** Leverages information theory to minimize the combined description length of the model and the data reconstruction error.