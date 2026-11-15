# Generative Adversarial Networks (GANs)

**Definition and Mathematical Foundation**
GANs are defined as a zero-sum game between two players: a generator ($\mu_G$) and a discriminator ($\mu_D$). Mathematically, the objective function aims for the generator to match the reference distribution ($\mu_{\text{ref}}$) and the discriminator to correctly distinguish between real and generated data. In practice, the generator maps a latent space to the true data distribution, while the discriminator evaluates these candidates.

**Mathematical Properties and Training**
The original GAN game theory proves that the optimal discriminator computes the Jensen-Shannon divergence between $\mu_{\text{ref}}$ and $\mu_G$. The unique equilibrium point dictates that the generator perfectly mimics the reference distribution ($\mu_G = \mu_{\text{ref}}$), and the discriminator outputs $1/2$ deterministically. Training stability is a major issue due to restricted strategy sets, leading to unstable convergence. Common challenges include mode collapse, where the generator fails to capture the diversity of the input data, and vanishing gradients when the discriminator becomes too proficient.

**Variants and Related Models**
Numerous variants exist to address GAN limitations. Conditional GANs allow generation based on specific class labels ($c$). Wasserstein GANs (WGANs) replace the JS divergence with the Wasserstein metric, providing smoother gradients and mitigating mode collapse. Architecturally, variants include DCGAN, SAGAN, and Transformer GANs. CycleGAN enables unpaired domain translation by adding a cycle consistency loss.

**Evaluation and Advanced Topics**
GANs are typically evaluated using metrics like Inception Score (IS) or Fréchet Inception Distance (FID). More complex models address training issues: the Two Time-Scale Update Rule (TTUR) stabilizes convergence, and data augmentation can be used, provided the transformation kernel is invertible.

# Transformer (Deep Learning)

**History and Core Mechanism**
Transformers originated from sequence-to-sequence (seq2seq) models that used RNNs (like LSTM) and suffered from bottleneck issues when processing long sequences. The breakthrough came in 2017 with "Attention is all you need," which replaced recurrence entirely with attention mechanisms, allowing for massive parallelization.

**Architecture Components**
The fundamental architecture relies on tokenizers (often using subword tokenizers like BPE), an embedding layer, and stacked Transformer layers. Each layer contains multihead attention and a feedforward network. Attention calculates relevance between tokens using scaled dot-product attention, which is then augmented by multiple heads to capture different types of relevance. Positional encoding provides the model with sequence order information.

**Variations and Applications**
Architectures are classified based on usage: Encoder-only (like BERT, used for text representation), Decoder-only (like GPT series, used for text generation), and Encoder-Decoder (like T5, used for sequence transduction). Applications span NLP (translation, summarization), computer vision (Vision Transformers, ViT), and multimodal tasks. Advanced techniques include RoPE and ALiBi for positional encoding, and FlashAttention for efficient GPU implementation.

# Convolutional Neural Networks (CNNs)

**Architecture and Principles**
CNNs are designed to exploit the spatial locality inherent in grid-like data like images, overcoming the curse of dimensionality of fully connected networks. Their key features are: **Local Connectivity** (receptive fields), **Shared Weights** (filters applied across the entire input), and **Pooling** (down-sampling to reduce parameters and grant local translational invariance).

**Building Blocks and Training**
Core building blocks include convolutional layers (which produce feature maps), pooling layers (Max or Average pooling), and fully connected layers for final classification. ReLU is a common, fast activation function. Training involves optimizing filters and biases, often using backpropagation. Hyperparameters like stride, padding, and filter size control the network's behavior.

**Applications and Enhancements**
CNNs excel in image recognition (e.g., AlexNet, GoogLeNet) and have been adapted for video analysis and time series. Regularization techniques like Dropout and L2 weight decay prevent overfitting. Further advancements include Depthwise Separable convolutions and specialized pooling methods like Channel Max Pooling.

# Vision Transformer (ViT)

**Overview and Mechanism**
ViT adapts the Transformer architecture for computer vision. The process involves splitting an input image ($\mathbb{R}^{H\times W\times C}$) into sequential, square-shaped patches. These patches are linearly projected ("patch embedding") and combined with positional encodings before being fed into a standard encoder-only Transformer block.

**Variants and Comparison with CNNs**
ViT's self-attention mechanism allows it to capture long-range dependencies across image patches, unlike CNNs which rely on local receptive fields. Variants like Swin Transformer use "shifted windows" to emulate CNN-like local processing for efficiency. Masked Autoencoders (MAE) and DINO leverage self-supervision by reconstructing masked patches or distilling knowledge from past states, respectively.

**Inductive Bias Differences**
CNNs have a strong inductive bias towards locality and translation equivariance through shared kernels. ViT, by relying on attention, is generally more data-hungry but shows robustness to input distortions and can capture global relationships effectively.

# Wasserstein GAN (WGAN)

**Motivation and Theory**
WGAN addresses the instability of standard GANs, which rely on the Jensen-Shannon divergence ($D_{JS}$). Since $D_{JS}$ can be infinite when the generator distribution ($\mu_G$) is disjoint from the reference distribution ($\mu_{\text{ref}}$) (common in high-dimensional data), the gradient vanishes.

**WGAN Definition and Objective**
WGAN replaces the objective with the Earth Mover's Distance (Wasserstein distance, $W_1$), defined via the Kantorovich-Rubenstein duality. The discriminator (now called the critic) is restricted to have a bounded Lipschitz norm ($\|D\|_L \leq K$), and the objective becomes $L_{WGAN} = \mathbb{E}_{x\sim \mu_G}[D(x)] - \mathbb{E}_{x\sim \mu_{\text{ref}}}[D(x)]$.

**Training Techniques**
To enforce the Lipschitz constraint, several methods are used for training the critic: **Weight Clipping** (clamping weights to $[-c, c]$), **Spectral Normalization** (constraining the spectral radius of weight matrices), or the **Gradient Penalty** (adding a penalty term to enforce the norm constraint).

# Autoencoder (AE)

**Mathematical Principles and Operation**
An Autoencoder (AE) is a system composed of an encoder ($E_{\phi}: \mathcal{X} \to \mathcal{Z}$) and a decoder ($D_{\theta}: \mathcal{Z} \to \mathcal{X}$), where the goal is to reconstruct the input $x$ from its compressed latent code $z = E_{\phi}(x)$. The training minimizes a reconstruction loss $L(\theta, \phi) = \mathbb{E}_{x\sim \mu_{\text{ref}}}[d(x, D_{\theta}(E_{\phi}(x)))]$.

**Interpretation and Variations**
In an ideal **undercomplete** AE, the code space $\mathcal{Z}$ is smaller than the input space $\mathcal{X}$, forcing the model to learn the most essential features for data compression.
Variations include:
*   **Variational Autoencoder (VAE):** Models the latent space as a distribution ($q_{\phi}(z|x)$) rather than a fixed vector, maximizing the Evidence Lower Bound (ELBO) by balancing reconstruction error and KL divergence. The reparameterization trick is crucial for gradient flow in VAEs.
*   **Denoising Autoencoder (DAE):** Trains by corrupting inputs ($\mu_T$) and forcing the network to recover the original, clean message.
*   **Sparse Autoencoder (SAE):** Enforces sparsity in the latent code by adding a regularization term (e.g., KL divergence or L1 loss).

**Applications**
AEs are used for dimensionality reduction (related to PCA), information retrieval, and anomaly detection (high reconstruction error indicates an anomaly). Deep AEs improve performance by exponentially reducing computational cost and data needs.

# Recurrent Neural Network (RNN)

**History and Fundamental Concept**
RNNs are networks featuring recurrent connections, where the output or hidden state at time $t$ depends on the state and input from previous time steps. This allows them to maintain a "memory" of sequential data. Early models include Elman and Jordan networks, which introduced simple recurrent connections.

**Modern Architectures and Training**
Modern RNNs, primarily **LSTM** and **GRU**, overcome the vanishing gradient problem of standard RNNs using internal gating mechanisms (forget, input, output gates). Seq2seq models pair an encoder RNN with a decoder RNN for transduction tasks like machine translation. Training is typically done via Backpropagation Through Time (BPTT).

**Advanced Configurations and Uses**
Variations include Bidirectional RNNs (processing sequence forwards and backwards), Stacked RNNs (deep RNNs), and specialized types like Independent RNNs (IndRNN). More advanced concepts include Neural Turing Machines (NTMs) which couple RNNs with external memory, and Continuous-Time RNNs (CTRNNs) modeled using differential equations.

# Graph Neural Network (GNN)

**Architecture and Operation**
GNNs process data structured as graphs $G=(V,E)$. The core operation is **Message Passing**, where each node updates its feature representation ($\mathbf{h}_u$) by aggregating information ($\bigoplus$) received from its neighbors ($N_u$). The update function is typically $\mathbf{h}_u = \phi(\mathbf{x}_u, \bigoplus_{v \in N_u} \psi(\mathbf{x}_u, \mathbf{x}_v, \mathbf{e}_{uv}))$.

**Layer Types and Limitations**
GNNs comprise three main stages: permutation-equivariant message passing layers, local pooling (to coarsen the graph), and global pooling (readout layer for fixed-size representation). A limitation is that GNNs cannot distinguish all non-isomorphic graphs (Weisfeiler–Leman test).
Variants include:
*   **Graph Convolutional Networks (GCNs):** A spectral filter approximation applied to graph data.
*   **Graph Attention Networks (GATs):** Incorporate attention coefficients ($\alpha_{uv}$) to selectively weigh neighbor information.
*   **Graph Sequence Neural Networks (GGS-NN):** Extend GNNs to output sequences by using GRU cells for message passing.

# Other Specialized Networks

*   **Hopfield Network:** A recurrent network used for associative memory. It stores memories as local minima in an energy function, utilizing Hebbian learning rules. Modern variants (Dense Associative Memories) increase storage capacity beyond the linear scaling of classical models.
*   **Radial Basis Function (RBF) Network:** A three-layer network using RBF activation functions centered at specific points ($\mathbf{c}_i$). They are universal approximators, and training involves fixing centers (often via k-means) and then fitting linear weights.
*   **Variational Autoencoder (VAE):** A generative model that models the data likelihood by approximating the intractable posterior distribution $p_{\theta}(z|x)$ with $q_{\phi}(z|x)$. The objective is the Evidence Lower Bound (ELBO), balancing reconstruction error and KL divergence.
*   **Deep Q-Network (DQN):** Combines deep neural networks with Q-learning from reinforcement learning, enabling direct learning from high-dimensional sensory inputs (e.g., Atari games).
*   **Classifier Interpretability Methods:** Methods like **Class Activation Mapping (CAM)** and **Grad-CAM** use gradients or architectural features (like Global Average Pooling) to generate heatmaps, highlighting the input regions most relevant to a specific class prediction in CNNs.