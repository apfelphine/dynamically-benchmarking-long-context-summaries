An autoencoder is a neural network that learns to compress and reconstruct data. It consists of two main components: an encoder, which maps input data to a lower-dimensional latent space, and a decoder, which reconstructs the input from the latent representation. The goal is to minimize the reconstruction error between the original input and the reconstructed output. Autoencoders can be trained using various methods, including least-squares optimization and gradient descent. 

Key variations include:
- **Variational Autoencoders (VAEs)**: Use probabilistic modeling with latent variables, incorporating a prior distribution and noise, and aim to learn a generative model of data.
- **Sparse Autoencoders (SAEs)**: Encourage sparsity in the latent codes to improve feature learning.
- **Denoising Autoencoders (DAEs)**: Improve representation by training on corrupted inputs to reconstruct the original data.
- **Contractive Autoencoders (CAEs)**: Add a contractive regularization loss to prevent overfitting and ensure robustness to input perturbations.
- **Minimum Description Length (MDL) Autoencoders**: Optimize for efficient data representation by minimizing the total description length.

Autoencoders are widely applied in tasks such as dimensionality reduction, information retrieval, anomaly detection, image processing, and drug discovery. They have also been used in generative models and in creating efficient representations for downstream tasks.