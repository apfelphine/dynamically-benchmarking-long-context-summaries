## Autoencoder

An **autoencoder** is a neural network architecture consisting of an encoder that maps an input from a high-dimensional space to a lower-dimensional latent space (code), and a decoder that reconstructs the input from this code. Both encoder and decoder are typically multilayer perceptrons. The network is trained to minimize the reconstruction error, often using an L2 (Euclidean) loss between the original input and its reconstructed output, with the goal of learning efficient and meaningful representations of the data. Autoencoders can be undercomplete (latent space dimension smaller than input), encouraging compression, or overcomplete, where regularization or architectural choices are needed to prevent learning a trivial identity function.

Many **variations** exist, adapting the basic structure for specific purposes:
- *Variational Autoencoder (VAE)* models the latent space as a distribution and uses probabilistic encoders/decoders.
- *Sparse Autoencoder (SAE)* enforces sparsity in the encoded vectors, either by clamping activations or adding a regularization loss.
- *Denoising Autoencoder (DAE)* trains the model to recover uncorrupted data from corrupted inputs.
- *Contractive Autoencoder (CAE)* penalizes the sensitivity of representations to input perturbations.
- *Minimum Description Length Autoencoder (MDL-AE)* leverages information theory to compress representations.
- *Concrete Autoencoder* is designed for discrete feature selection.

Autoencoders are applied to dimensionality reduction, information retrieval, anomaly detection, image processing (denoising, super-resolution), feature learning, generative modeling, and more. Deep autoencoders (with multiple layers) provide better data compression and richer learned representations than shallow ones. Historically, linear autoencoders relate closely to principal component analysis (PCA), but nonlinear and deep variants capture more complex data structures. Key training strategies include regularization and joint training, and both supervised and unsupervised settings are common.