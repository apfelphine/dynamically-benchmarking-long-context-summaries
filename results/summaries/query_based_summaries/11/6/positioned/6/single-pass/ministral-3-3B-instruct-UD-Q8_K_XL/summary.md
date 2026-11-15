An **autoencoder** is a neural network architecture designed for unsupervised learning, consisting of an encoder and a decoder. The encoder compresses input data into a lower-dimensional latent representation (code), while the decoder reconstructs the original input from this latent space. The primary goal is to minimize the reconstruction error between the input and its reconstruction, effectively learning a compressed representation of the data.

Key components include:
- **Encoder**: A neural network that maps input data to a lower-dimensional latent space.
- **Latent Space**: A compressed representation of the input data, typically smaller in dimensionality than the input.
- **Decoder**: A neural network that reconstructs the original input from the latent representation.

Autoencoders are often used for:
- **Dimensionality Reduction**: Learning a lower-dimensional representation of high-dimensional data.
- **Feature Learning**: Extracting meaningful features from data, useful for tasks like anomaly detection and image compression.
- **Unsupervised Learning**: Training without labeled data, focusing on capturing intrinsic patterns in the data.

Variants include:
- **Variational Autoencoders (VAEs)**: Introduce probabilistic latent variables, allowing for more flexible and interpretable representations.
- **Sparse Autoencoders**: Encourage sparsity in the latent representations to focus on important features.
- **Denoising Autoencoders**: Train the network to reconstruct noisy versions of the input, learning robust representations.

Autoencoders are widely used in applications like image compression, anomaly detection, and feature extraction due to their ability to learn efficient, compact representations of data.