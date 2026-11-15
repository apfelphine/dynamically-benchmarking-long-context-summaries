# Summary on Autoencoder

An **autoencoder** is a neural network architecture designed to compress data by learning an encoder function \( E_{\phi}:\mathcal{X} \to \mathcal{Z} \) mapping inputs to a lower-dimensional latent space, and a decoder function \( D_{\theta}:\mathcal{Z} \to \mathcal{X} \) reconstructing the input from this representation. Typically, both encoder and decoder are multilayer perceptrons. Training involves minimizing a loss function that measures the reconstruction error between the input and its decoded output over a data distribution, often using the squared Euclidean distance and empirical datasets.

Autoencoders are primarily used for **dimensionality reduction**, compressing input while preserving essential information, and **information retrieval** by encoding data into compressed representations useful for tasks like semantic hashing. Variations include:

- **Variational Autoencoder (VAE):** Introduces a probabilistic formulation with latent variables modeled as distributions, trained by maximizing an evidence lower bound, supporting generative modeling.
- **Sparse Autoencoder (SAE):** Encourages sparsity in the latent code via hard k-sparsity or regularized sparsity losses, improving feature learning.
- **Denoising Autoencoder (DAE):** Trained to remove corruption/noise from input by reconstructing clean data, thereby learning robust representations.
- **Contractive Autoencoder (CAE):** Adds a regularization term penalizing the sensitivity of latent representations to input perturbations, promoting robustness.
- **Minimum Description Length Autoencoder (MDL-AE):** Combines reconstruction error with code length minimization from information theory.
- **Concrete Autoencoder:** Targets discrete feature selection using continuous relaxations allowing backpropagation through feature selection.

Deep autoencoders with multiple layers generally yield better compression and require less data to train compared to shallow variants. Training deep autoencoders often involves layerwise pretraining (e.g., via restricted Boltzmann machines) followed by fine-tuning through backpropagation.

The autoencoder concept originated in the 1980s as a nonlinear extension of PCA, with contributions that interpreted single-layer linear autoencoders as performing PCA, and extended this to nonlinear settings. Modern advancements have incorporated autoencoders into generative models and components of state-of-the-art systems.

Applications of autoencoders span dimensionality reduction, anomaly detection via reconstruction error, image processing tasks like compression and denoising, drug discovery, social media popularity prediction, neural machine translation, and communication systems where they help encode data resiliently.