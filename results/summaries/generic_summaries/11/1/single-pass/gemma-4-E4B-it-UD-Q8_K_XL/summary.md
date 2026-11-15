# Summary of Neural Network Architectures and Models

## Transformer (Deep Learning)
Transformers evolved from previous sequence models like RNNs and LSTMs, which struggled with long-sequence information due to sequential processing. The original 2017 Transformer introduced **self-attention** to enable parallel processing, although this incurs quadratic computation cost.

**Key Components and Concepts:**
*   **Architecture:** Primarily uses an **encoder-decoder** structure, but variants exist: **encoder-only** (e.g., BERT, used for embeddings/classification) and **decoder-only** (e.g., GPT series, used for text generation).
*   **Mechanism:** Core operations include **Tokenizers** (converting text to numerical sequences), **Embedding** and **Positional Encoding** (to inject sequence order), and **Transformer Layers** containing self-attention and feedforward networks.
*   **Attention:** Utilizes **Scaled Dot-Product Attention**, often implemented in **Multihead Attention** for parallelizing the concept of "relevance." **Masked attention** (causal masking) is used in decoders to enforce autoregressive behavior.
*   **Training:** Typically involves **pretrain-finetune** using self-supervised learning (e.g., language modeling, next-sentence prediction). Training stabilization methods include **learning rate warmup** or **layer normalization before** attention/FFN layers.
*   **Applications:** Successful across NLP (translation, summarization), and multimodal domains (Vision Transformer, image/video generation via DALL-E, Stable Diffusion).

## Convolutional Neural Network (CNN)
CNNs are designed to exploit the spatial locality in data (like images), mitigating the "curse of dimensionality" associated with fully connected layers.

**Key Components and Concepts:**
*   **Core Operations:** Involve **Convolutional layers**, which apply shared, small filters across the input to generate feature maps. **Pooling layers** (max or average) reduce data dimensions, offering local translational invariance. **Fully connected layers** are used for final classification.
*   **Distinguishing Features:** CNNs use **Local connectivity** (via receptive fields) and **Shared weights** (filters are replicated) to reduce parameters and achieve translational equivariance.
*   **Evolution and History:** Early concepts were inspired by the visual cortex (Hubel and Wiesel). Breakthroughs came with trainable architectures like the **Neocognitron** (Fukushima). Modern acceleration relied on **GPU implementations**, culminating in **AlexNet** (2012).
*   **Applications:** Widely used for image recognition (ImageNet), video analysis, and in generating synthetic data.

## Recurrent Neural Network (RNN)
RNNs are designed for sequential data processing, where the hidden state ($h_t$) acts as a memory of previous inputs.

**Key Components and Concepts:**
*   **Standard RNN:** Maps current input ($x_t$) and hidden state ($h_t$) to an output ($y_t$) and a new hidden state ($h_{t+1}$).
*   **Addressing Vanishing Gradient:** **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) use gating mechanisms to allow error signals to flow backward over long time steps, solving the vanishing gradient problem inherent in plain RNNs.
*   **Configurations:** Can be **Stacked** (deep RNNs), **Bidirectional** (processing sequence forward and backward, as in **ELMo**), or **Encoder-decoder** (sequence transduction, used heavily before Transformers).
*   **Training:** Uses **Backpropagation Through Time (BPTT)**. Techniques like **Teacher Forcing** stabilize autoregressive training.

## Generative Adversarial Network (GAN)
GANs are a zero-sum game between two neural networks: a Generator ($G$) and a Discriminator ($D$).

**Key Components and Concepts:**
*   **The Game:** $G$ attempts to map random noise from a latent space ($\mu_Z$) to match the real data distribution ($\mu_{ref}$), while $D$ attempts to distinguish between real and generated samples.
*   **Mathematical Foundation:** The original GAN objective is linked to the **Jensen-Shannon Divergence** ($D_{JS}$).
*   **Training Issues:** GANs often suffer from **mode collapse** (where $G$ only generates a subset of $\mu_{ref}$) and **vanishing gradients** (if $D$ becomes too good).
*   **Variants:**
    *   **Wasserstein GAN (WGAN):** Replaces the JS divergence with the **Wasserstein distance** ($W_1$), providing a smoother loss landscape and better gradients. Training stability is enforced by bounding the discriminator's Lipschitz norm (using methods like **weight clipping** or **spectral normalization**).
    *   **Conditional GAN (cGAN):** Allows generation based on auxiliary information (e.g., class labels).
    *   **CycleGAN:** Enables image translation between two domains without requiring paired training data by using a **cycle consistency loss**.
    *   **StyleGAN:** A family of architectures (including Progressive GAN) designed for high-resolution image generation, incorporating progressive growing and style latent vector manipulation.

## Autoencoder (AE) and Variational Autoencoder (VAE)
Autoencoders are unsupervised models designed for dimensionality reduction and feature learning by learning to reconstruct their input.

**Key Components and Concepts:**
*   **Architecture:** Consists of an **Encoder** ($E_{\phi}$) mapping input ($\mathcal{X}$) to a compressed latent code ($\mathcal{Z}$), and a **Decoder** ($D_{\theta}$) reconstructing the input from the code.
*   **Training:** Trained to minimize a **reconstruction quality function** ($L = \mathbb{E}[d(x, D(E(x)))]$). **Undercomplete** AEs force compression by having $\dim(\mathcal{Z}) < \dim(\mathcal{X})$.
*   **Variational Autoencoder (VAE):** A probabilistic generative model. Instead of mapping $x \to z$ deterministically, the encoder maps $x$ to parameters ($\mu, \sigma$) of a distribution $q_{\phi}(z|x)$. Training optimizes the **Evidence Lower Bound (ELBO)**, balancing reconstruction error against the divergence ($\text{KL-D}$) between $q_{\phi}(z|x)$ and a prior distribution $p_{\theta}(z)$.
*   **Variations:** **Denoising Autoencoders (DAE)** corrupt the input during training. **Sparse AEs (SAE)** enforce sparsity in the latent code. **Contractive AEs (CAE)** penalize changes in the code resulting from infinitesimal input variations, promoting robust features.

## Graph Neural Network (GNN)
GNNs extend CNN concepts to data structured as graphs (nodes and edges).

**Key Components and Concepts:**
*   **Message Passing:** The core primitive, where nodes update their representations ($\mathbf{h}_u$) by aggregating **messages** ($\psi$) received from their neighbors ($N_u$). Stacking layers increases the receptive field.
*   **Architectures:**
    *   **Graph Convolutional Network (GCN):** A spectral approximation of localized filters on graphs.
    *   **Graph Attention Network (GAT):** Incorporates attention coefficients ($\alpha_{uv}$) to weight neighbor messages, allowing nodes to focus on more important neighbors.
*   **Pooling:** Used for downsampling the graph (coarsening) via **Local Pooling** strategies, such as **Top-k pooling** or **Self-attention pooling**.
*   **Challenges:** GNNs are limited by the **Weisfeiler–Leman Test**, meaning they cannot distinguish all non-isomorphic graphs. **Heterophily** (connecting dissimilar nodes) is a noted challenge.

## Other Architectures and Techniques
*   **Radial Basis Function (RBF) Networks:** Three-layer networks using a radial basis function (e.g., Gaussian) in the hidden layer to approximate a continuous function. They can be trained using two steps: unsupervised center selection followed by linear weight fitting. Normalization improves accuracy.
*   **Hopfield Networks:** Models for associative memory, where stored patterns become stable fixed-point attractors in the network's energy landscape. Modern variations (**Dense Associative Memories**) use higher-order interactions to achieve super-linear memory capacity.
*   **Attention Visualization:** Techniques like **Class Activation Mapping (CAM)** and **Grad-CAM** use gradients or global average pooling to produce heatmaps, highlighting which input regions were most important for a CNN's prediction. **Guided Grad-CAM** fuses this with guided backpropagation for higher resolution.
*   **Low-Rank/Efficient Attention:** Techniques like **FlashAttention** and **Multi-Query/Grouped-Query Attention (MQA/GQA)** optimize the quadratic computational cost of attention for faster GPU inference.