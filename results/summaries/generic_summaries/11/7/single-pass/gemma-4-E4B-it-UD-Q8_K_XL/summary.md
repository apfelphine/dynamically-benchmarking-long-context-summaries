# Graph Neural Networks (GNNs)
GNNs comprise fundamental layers: **Permutation equivariant layers** (implemented via message passing, where nodes aggregate messages from neighbors, increasing receptive field by one hop), **Local pooling layers** (downsampling the graph, similar to CNN pooling), and **Global pooling layers** (readout layers providing a fixed-size, permutation-invariant graph representation). GNN expressiveness is theoretically bounded by the Weisfeiler–Leman Graph Isomorphism Test.

**Message Passing Neural Networks (MPNNs)** formalize this, where node representations update via aggregation of neighbor messages. Stacking MPNN layers increases the receptive field, but this can lead to issues like **oversmoothing** or **oversquashing**, which can be countered with skip connections or gated rules. Specific GNN types include **Graph Convolutional Networks (GCNs)**, which approximate a localized spectral filter and are limited in handling multidimensional edge features, and **Graph Attention Networks (GATs)**, which incorporate attention coefficients ($\alpha_{uv}$) to focus on important local information, capable of handling edge features.

Local pooling strategies include **Top-k pooling**, which retains the nodes with the highest projection scores, and **Self-attention pooling**, an extension that incorporates graph features and topology into the selection process.

# Variational Autoencoders (VAEs)
A VAE is a generative model employing an **encoder** neural network to map data points to a low-dimensional latent space, and a **decoder** neural network to reconstruct the input from this latent space. VAEs optimize the **Evidence Lower Bound (ELBO)**, which balances minimizing the **reconstruction error** and minimizing the **Kullback–Leibler divergence (KL-D)** between the approximated posterior ($q_{\phi}(z|x)$) and the prior ($p_{\theta}(z)$).

Optimization relies on the **reparameterization trick**, which allows gradients to flow through the stochastic sampling process by rewriting the latent variable $z$ as $z = \mu_{\phi}(x) + L_{\phi}(x)\epsilon$. Variations include **$\beta$-VAE** (which weights the KL-D for disentanglement) and **Conditional VAEs (CVAEs)**, which inject label information into the latent space. Statistical distance VAEs replace the KL-D with other distances like Wasserstein or MMD.

# Autoencoders (AEs)
An AE consists of an **encoder** ($E_{\phi}: \mathcal{X} \to \mathcal{Z}$) and a **decoder** ($D_{\theta}: \mathcal{Z} \to \mathcal{X}$), trained to minimize the reconstruction loss, $L(\theta, \phi) = \mathbb{E}_{x \sim \mu_{ref}}[d(x, D_{\theta}(E_{\phi}(x)))]$. Undercomplete AEs compress data, while overcomplete ones can potentially learn the identity function.

Variations include:
*   **Variational Autoencoders (VAEs):** Introduce a probabilistic framework with a latent space composed of distributions.
*   **Sparse Autoencoders (SAEs):** Enforce sparsity in the latent codes, either via clamping or sparsity regularization loss.
*   **Denoising Autoencoders (DAEs):** Enhance robustness by training on intentionally corrupted inputs.
*   **Contractive Autoencoders (CAEs):** Regularize the encoder to ensure small input variations map to small variations in the code via a contractive loss on the Jacobian.
*   **Minimum Description Length Autoencoders (MDL-AEs):** Optimize based on minimizing the combined length of the code and the reconstruction error.

# Generative Adversarial Networks (GANs)
A GAN is a zero-sum game between two players: a **generator** ($\mu_G$) which aims to match the reference data distribution ($\mu_{ref}$), and a **discriminator** ($D$) which tries to distinguish real from generated data. The original GAN minimizes the **Jensen–Shannon divergence** between the two distributions.

Training stability issues include **mode collapse** (generator only producing a subset of the data modes) and **vanishing gradients** (if the discriminator becomes too effective). Solutions include the **Two Time-Scale Update Rule (TTUR)**. **Wasserstein GANs (WGANs)** replace the Jensen-Shannon divergence with the **Wasserstein distance**, which provides smoother and more meaningful gradients, and training is stabilized by enforcing a bounded Lipschitz norm on the discriminator through techniques like **weight clipping**, **spectral normalization**, or **gradient penalty**.

# Convolutional Neural Networks (CNNs)
CNNs are designed to exploit the spatial locality and translational equivariance in grid-like data (like images). Key architectural elements include **Convolutional layers**, which apply shared, localized kernels (filters) across the input to generate feature maps; **Pooling layers** (max or average) for downsampling and achieving local translational invariance; and **Fully connected layers** for final classification.

CNNs address the curse of dimensionality of MLPs by parameter sharing. Hyperparameters like **stride** and **padding** control the output volume size and receptive field size. Regularization methods include **Dropout**, **DropConnect**, and **L2/L1 weight decay**. Recent advances involve techniques like **dilated convolutions** to expand receptive fields without increasing parameters.

# Recurrent Neural Networks (RNNs)
RNNs process data sequentially, using a **hidden vector** ($h_t$) as "memory" to retain partial records of past inputs. The core mechanism maps an input ($x_t$) and the previous hidden state ($h_t$) to an output ($y_t$) and a new hidden state ($h_{t+1}$).

To overcome the **vanishing/exploding gradient problem**, modern RNNs use sophisticated gating mechanisms like **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRUs)**. Architectures include **Stacked RNNs** (deep RNNs), **Bidirectional RNNs (BiRNNs)** (processing sequence forward and backward), and **Encoder-Decoder** configurations for sequence transduction. Training typically involves **Backpropagation Through Time (BPTT)**, though optimization can also use global methods like genetic algorithms.

# Transformer (deep learning)
The Transformer architecture, introduced in "Attention Is All You Need," revolutionized sequence modeling by completely removing recurrence and relying solely on **self-attention** to process all tokens in parallel, resulting in quadratic time complexity $O(N^2)$ in sequence length $N$.

The structure involves **Tokenizers** (converting text to numerical tokens), **Embedding** layers, and repeated **Transformer Layers**, which consist of **Multihead Attention** and a **Feedforward Network (FFN)**, augmented by residual connections and layer normalization. **Causal masking** is used in decoders to enforce autoregressive generation. Key variations include:
*   **Encoder-only** (e.g., BERT, used for representation learning).
*   **Decoder-only** (e.g., GPT, used for text generation).
*   **Encoder-Decoder** (original setup, used for machine translation).

Efficiency improvements include **FlashAttention-2** (optimizing GPU memory movement), **KV caching** for inference speedup, and sub-quadratic methods like **Reformer** or **Swin Transformer** (for vision).

# Vision Transformer (ViT)
ViT adapts the Transformer for computer vision by treating image patches as sequences of tokens. The input image ($\mathbb{R}^{H \times W \times C}$) is split into patches, each passed through a linear operator to create a **patch embedding**, which is then augmented with **positional encoding**. These embeddings are fed into a BERT-like encoder-only Transformer.

Variants have improved efficiency and performance:
*   **Swin Transformer** uses convolution-like **shifted windows** for localized attention.
*   **Masked Autoencoder (MAE)** uses a two-ViT structure to reconstruct masked image patches for unsupervised pretraining.
*   **DINO** uses self-distillation (teacher-student) for self-supervision.
*   **TimeSformer** applies factorized attention for video understanding.

ViT excels at capturing long-range relationships between image parts, contrasting with CNNs' reliance on local receptive fields.

# Wasserstein GAN (WGAN)
WGAN modifies the standard GAN objective by replacing the Jensen–Shannon divergence with the **Wasserstein metric** ($W_1$). This metric is calculated via the **Kantorovich-Rubenstein duality**, where the discriminator acts as a **critic** whose Lipschitz-norm bounded function estimates the distance between the generator and reference distributions.

To enforce the Lipschitz constraint ($\|D\|_L \leq K$), several training methods are employed: **weight clipping** (restricting weights to $[-c, c]$), **spectral normalization** (constraining weights by dividing by their spectral radius), and the **gradient penalty** (adding a term that forces the discriminator's gradient to remain close to a constant $a$). WGANs are noted for providing more stable gradients and mitigating mode collapse compared to original GANs.

# Hopfield Network
The Hopfield network is an associative memory model, functioning as a recurrent network with discrete or continuous states, defined by an **energy function** ($E$). Training involves setting memory vectors as local minima in this energy landscape, typically using the **Hebbian learning rule** ($\text{w}_{ij} = \frac{1}{n}\sum \epsilon_i^\mu\epsilon_j^\mu$).

The network retrieves memories by converging to the closest local minimum when presented with a corrupted (partially active) input state. **Dense Associative Memories** are generalizations that break the linear scaling of memory capacity by introducing higher-order interactions, allowing for exponential memory capacity.

# Radial Basis Function Network (RBFN)
RBFN networks are typically three-layered: input, a hidden layer with a radial basis function activation, and a linear output layer. The radial basis function measures the distance between the input $\mathbf{x}$ and a hidden neuron's center $\mathbf{c}_i$.

The output is a weighted sum of these basis functions. The network can be **normalized** by dividing by the sum of the basis functions, which has theoretical grounding in approximating conditional expectations from stochastic data. Training is usually a two-step process: first, fixing the centers $\mathbf{c}_i$ (often via k-means clustering), and second, fitting the linear output weights $a_i$ using least squares or gradient descent.

# Comparison Summary
| Model | Primary Function | Key Mechanism | Scalability/Complexity |
| :--- | :--- | :--- | :--- |
| **CNN** | Image feature extraction | Shared weights, local connectivity, pooling | Parameter efficiency; local features |
| **GNN** | Graph data processing | Message passing (aggregation) | Relational structure; expressive limits |
| **RNN** | Sequential data modeling | Hidden state memory ($h_t$) | Sequence dependency; vanishing gradients |
| **Transformer**| Sequence/Token modeling | Self-attention, parallel processing | Quadratic complexity in sequence length ($N^2$) |
| **VAE** | Generative modeling | Encoder/Decoder, ELBO optimization | Implicit density estimation; probabilistic |
| **GAN** | Generative modeling | Adversarial zero-sum game | Training instability (mode collapse, vanishing gradients) |
| **RBFN** | Function approximation/Interpolation| Distance-based hidden layer activation | Simple, local basis function fitting |
| **Hopfield**| Associative memory | Energy function minimization, Hebbian learning | Stores patterns as energy minima; capacity limited |