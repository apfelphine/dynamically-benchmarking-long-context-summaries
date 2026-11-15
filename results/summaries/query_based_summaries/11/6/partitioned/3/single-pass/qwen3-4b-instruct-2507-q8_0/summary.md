An **autoencoder** is a neural network that learns to compress data into a latent representation and reconstruct it, with the goal of minimizing reconstruction error. It consists of an encoder that maps input data to a code and a decoder that reconstructs the input from the code. The model is trained to perform near-perfect reconstruction, typically using a dataset with a known reference distribution and a reconstruction quality function like L2 loss.

Key applications include:
- **Dimensionality reduction**: Autoencoders reduce data dimensions while preserving structure, often outperforming PCA due to their non-linear capabilities. A 2006 study showed that a deep autoencoder with a bottleneck of 30 neurons achieved lower reconstruction error than PCA and produced more interpretable clusters.
- **Information retrieval**: Used in semantic hashing, where low-dimensional binary codes are generated for efficient search.
- **Anomaly detection**: By learning normal data patterns, autoencoders flag anomalies through high reconstruction errors. However, some models may fail to detect anomalies due to the nature of the training data.
- **Image processing**: Applied in image denoising, compression (outperforming JPEG 2000), and super-resolution.
- **Drug discovery**: Variational autoencoders generated molecules validated in mice.
- **Popularity prediction**: Stacked autoencoders predicted social media post popularity.
- **Machine translation**: Used in neural machine translation (NMT), though largely replaced by transformer networks.
- **Communication systems**: Used to encode data robustly against channel impairments.

Variations enhance performance:
- **Denoising autoencoders (DAE)**: Train on corrupted inputs to learn robust representations.
- **Contractive autoencoders (CAE)**: Add regularization to make latent representations stable to input perturbations.
- **Sparse autoencoders (SAE)**: Encourage sparsity in latent codes to improve feature learning.
- **Variational autoencoders (VAE)**: Generate data by modeling the latent space as a probability distribution, enabling generative modeling.

Deep autoencoders offer advantages such as better compression, reduced training data needs, and improved feature learning compared to shallow models. Training often uses techniques like deep belief networks or joint training, with regularization strategies improving performance. The architecture has evolved from early models to modern variants like VAEs and DAEs, which are integral components in advanced AI systems.