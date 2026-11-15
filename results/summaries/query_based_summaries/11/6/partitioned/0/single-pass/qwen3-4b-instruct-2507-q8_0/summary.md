An autoencoder is a neural network that learns to compress data into a latent space and reconstruct it. It consists of an encoder that maps input data to a code (latent representation) and a decoder that reconstructs the input from the code. The model is trained to minimize reconstruction error, with the optimal autoencoder achieving near-perfect reconstruction as defined by a quality function.

The autoencoder is typically defined with two sets: the space of input messages ${\mathcal{X}}$ and the space of latent codes ${\mathcal{Z}}$, where ${\mathcal{X}}$ is usually larger than ${\mathcal{Z}}$. The encoder and decoder are parametrized functions, often implemented as multilayer perceptrons (MLPs). Training involves minimizing a loss function over a dataset, typically using mean squared error or L2 loss, with the goal of learning a compressed representation.

Key variations include:
- **Variational Autoencoder (VAE)**: Uses probabilistic modeling with a latent prior (often Gaussian) and a variational posterior. It optimizes the Evidence Lower Bound (ELBO), balancing reconstruction accuracy and latent space regularity via Kullback–Leibler divergence.
- **Denoising Autoencoder (DAE)**: Trains the model to reconstruct clean data from noisy inputs, improving robustness to noise.
- **Sparse Autoencoder (SAE)**: Encourages sparsity in the latent code, where only a few units are active, by adding sparsity regularization.
- **Contractive Autoencoder (CAE)**: Adds a regularization term to penalize large changes in the latent code with small input changes, promoting stability.
- **Minimum Description Length (MDL) Autoencoder**: Minimizes the total description length of the data and latent code, promoting compact and efficient representations.
- **Concrete Autoencoder**: Forces the latent space to contain only a user-specified number of features, using continuous relaxation for gradient flow.

Deep autoencoders (with multiple layers) offer advantages over shallow ones, including better compression, reduced data requirements, and improved feature learning. Training deep autoencoders often uses pretraining with restricted Boltzmann machines followed by fine-tuning via backpropagation.

Applications span dimensionality reduction, anomaly detection, image processing (e.g., denoising, super-resolution), medical imaging, drug discovery, and machine translation. Autoencoders are foundational in generative models like VAEs and are used in systems such as Stable Diffusion and DALL-E.