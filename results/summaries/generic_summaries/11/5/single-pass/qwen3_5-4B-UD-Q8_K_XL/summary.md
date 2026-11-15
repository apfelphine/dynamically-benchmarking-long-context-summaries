# Summary of Neural Network Architectures

## Radial Basis Function Networks (RBFN)
Radial basis function networks consist of an input layer, a hidden layer utilizing non-linear radial basis functions (typically Gaussian), and a linear output layer. The output is a weighted sum of basis functions centered at specific vectors, where the activation depends on the Euclidean distance between the input and the center. These networks are universal approximators, capable of modeling any continuous function on a compact subset given sufficient hidden neurons.

Training typically involves a two-step algorithm: first, unsupervised determination of center vectors via random sampling or clustering (e.g., k-means), and second, supervised linear fitting of the output weights to minimize a loss function like least squares. Variations include normalized architectures (which yield better accuracy, especially with high input dimensionality) and local linear models that incorporate gradient terms for improved approximation.

## Autoencoders and Variations
Autoencoders are neural networks designed to learn a compressed representation of data by encoding input into a lower-dimensional latent space and decoding it back to the original space. The goal is to minimize the reconstruction error, often formulated as a least-squares optimization. Undercomplete autoencoders act as dimensionality reduction tools, learning features that capture the essential structure of the data distribution, while overcomplete ones may learn trivial identity mappings unless constrained.

Variations address specific limitations of standard autoencoders:
*   **Variational Autoencoders (VAE):** Introduce probabilistic latent variables, optimizing a lower bound (ELBO) to ensure a continuous, smooth latent space suitable for generative modeling.
*   **Sparse Autoencoders:** Enforce sparsity in the latent code to improve interpretability and performance on classification tasks.
*   **Denoising Autoencoders:** Train on corrupted inputs to learn robust representations resistant to noise.
*   **Contractive Autoencoders:** Add regularization to the loss function to make the learned representation invariant to small input perturbations.

## Recurrent Neural Networks (RNN)
Recurrent neural networks are designed to process sequential data by maintaining a hidden state that acts as memory, allowing information from previous time steps to influence current outputs. Standard RNNs suffer from the vanishing gradient problem, which hinders the learning of long-term dependencies.

Modern architectures address this through specialized structures:
*   **LSTM (Long Short-Term Memory):** Utilizes gating mechanisms (forget, input, output gates) to control information flow, enabling the network to retain information over thousands of time steps.
*   **GRU (Gated Recurrent Unit):** A simplified version of LSTM with fewer parameters but comparable performance.
*   **Bidirectional RNNs:** Process sequences in both forward and backward directions, concatenating results to provide full contextual information for each token.
Training often employs Backpropagation Through Time (BPTT), while advanced techniques like Teacher Forcing and Connectionist Temporal Classification (CTC) facilitate effective sequence-to-sequence learning.

## Graph Neural Networks (GNN)
Graph Neural Networks are designed to learn from data structured as graphs, where nodes and edges represent entities and relationships. The core mechanism is the **Message Passing Neural Network (MPNN)**, where node representations are updated by aggregating information from neighboring nodes. This process is repeated across layers, effectively expanding the receptive field of each node.

Common implementations include:
*   **Graph Convolutional Networks (GCN):** Use spectral or spatial convolutions to aggregate neighbor features.
*   **Graph Attention Networks (GAT):** Assign learnable attention coefficients to neighbors, allowing the model to focus on the most relevant parts of the graph.
GNNs are permutation-equivariant, meaning the output changes predictably when the input graph nodes are reordered. They are widely used in social networks, molecular modeling (e.g., protein folding), and combinatorial optimization.

## Transformers
Transformers are deep learning architectures based on the **self-attention mechanism**, which allows models to weigh the importance of different parts of the input sequence simultaneously. Unlike RNNs, transformers process all tokens in parallel, eliminating sequential bottlenecks but introducing quadratic computational complexity regarding sequence length.

Key components include:
*   **Encoder-Decoder Architecture:** Used for tasks like machine translation, where an encoder captures input context and a decoder generates output sequences.
*   **Multi-Head Attention:** Allows the model to attend to information from different representation subspaces at different positions.
*   **Positional Encoding:** Since attention is permutation-invariant, positional information is added to input vectors to preserve sequence order.
Transformers have revolutionized natural language processing (e.g., BERT, GPT) and vision (e.g., Vision Transformers), often surpassing traditional RNNs and CNNs in performance.

## Generative Adversarial Networks (GAN)
Generative Adversarial Networks consist of two competing neural networks: a **generator** that creates synthetic data and a **discriminator** that attempts to distinguish real data from the generator's output. This zero-sum game drives the generator to produce increasingly realistic samples that fool the discriminator.

Key challenges and solutions in GAN training include:
*   **Mode Collapse:** The generator producing limited variety of outputs, often solved by techniques like the Two Time-Scale Update Rule (TTUR) or the Wasserstein GAN (WGAN).
*   **Vanishing Gradients:** Occurs when the discriminator is too good; WGAN addresses this by using the Wasserstein distance, which provides smoother gradients.
*   **Variants:** Conditional GANs generate specific classes, CycleGAN performs unpaired domain translation, and StyleGAN offers fine-grained control over generated image styles.

## Wasserstein GAN (WGAN)
The Wasserstein GAN (WGAN) is an improvement over the original GAN formulation that replaces the Jensen-Shannon divergence with the **Wasserstein distance** (Earth Mover's Distance). This metric measures the cost of transporting mass from the generator's distribution to the real data distribution.

WGAN introduces strict constraints on the discriminator (now called a "critic") to ensure the Lipschitz continuity of the loss function. This is typically achieved via **weight clipping**, **spectral normalization**, or a **gradient penalty**. These modifications stabilize training, prevent mode collapse, and allow the generator to converge to a Nash equilibrium where the generated distribution closely matches the real data distribution, even for high-dimensional inputs like images.