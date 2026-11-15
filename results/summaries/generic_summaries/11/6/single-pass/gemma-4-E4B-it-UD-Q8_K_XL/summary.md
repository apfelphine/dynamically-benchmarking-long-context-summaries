# Generative Models and Neural Architectures

## Autoencoders (AE)
Autoencoders consist of an encoder ($E_\phi: \mathcal{X} \to \mathcal{Z}$) and a decoder ($D_\theta: \mathcal{Z} \to \mathcal{X}$), typically implemented as Multilayer Perceptrons (MLPs). Training involves minimizing a reconstruction loss $L(\theta, \phi) = \mathbb{E}_{x\sim \mu_{ref}}[d(x, D_\theta(E_\phi(x)))]$. Undercomplete AEs compress data, while overcomplete ones risk learning the identity function.

Variations include:
*   **Variational Autoencoders (VAEs):** Generative models that model data as a distribution $p_\theta(x)$ by approximating the true posterior $p_\theta(z|x)$ with a tractable distribution $q_\phi(z|x)$, optimized using the Evidence Lower Bound (ELBO).
*   **Sparse Autoencoders (SAE):** Encourage codes ($E_\phi(x)$) to be sparse using regularization losses (e.g., KL divergence or L1/L2 norms).
*   **Denoising Autoencoders (DAE):** Enhance representation by intentionally corrupting inputs during training and forcing recovery of the original message.
*   **Contractive Autoencoders (CAE):** Enforce stability by minimizing the expected squared Frobenius norm of the encoder's Jacobian, making the code robust to small input variations.
*   **Minimum Description Length AE (MDL-AE):** Minimizes a total description length comprising latent representation size and reconstruction error.

## Graph Neural Networks (GNNs)
GNN architectures involve three fundamental layers: **permutation equivariant layers** (implemented via message passing), **local pooling** for downsampling, and **global pooling** (readout) for fixed-size graph representation.

**Message Passing Neural Networks (MPNNs)** define how nodes update representations by aggregating messages from neighbors. Stacking MPNN layers can lead to oversmoothing or oversquashing. Specific GNNs include:
*   **Graph Convolutional Network (GCN):** A generalization of CNNs to graphs, using a first-order spectral filter approximation.
*   **Graph Attention Network (GAT):** Integrates attention mechanisms, allowing the network to assign importance coefficients ($\alpha_{uv}$) to neighbors, making it a generalization of GCNs.
*   **Gated Graph Sequence Neural Network (GGS-NN):** Extends GNNs to output sequences by incorporating a Gated Recurrent Unit (GRU) cell into the message passing framework.

## Recurrent Neural Networks (RNNs)
RNNs use a hidden state ($h_t$) to maintain a memory of past input-output pairs, processing data sequentially over time steps. Architectures include:
*   **Standard RNN:** Maps input ($x_t$) and hidden state ($h_t$) to output ($y_t$) and the next hidden state ($h_{t+1}$).
*   **Stacked RNN:** Composed of multiple RNNs where the output of one layer feeds into the next.
*   **Bidirectional RNN (biRNN):** Processes input sequences both forward and backward, concatenating the results to capture context before and after a token.
*   **Long Short-Term Memory (LSTM) & Gated Recurrent Unit (GRU):** Solutions to the vanishing gradient problem, allowing memory retention over many time steps.
*   **Encoder-Decoder:** Uses two RNNs (encoder and decoder) for sequence transduction tasks like machine translation, often augmented with attention mechanisms.

## Transformer Architecture
Transformers revolutionized sequence modeling by abandoning recurrence in favor of attention, allowing for massive parallelization.

The core components are:
*   **Tokenization & Embedding:** Text is converted into numerical tokens, which are then mapped to high-dimensional embedding vectors.
*   **Positional Encoding:** Since there is no recurrence, fixed positional encodings (like sinusoidal functions or RoPE) are added to inject information about token order.
*   **Transformer Layers:** Alternating **Multihead Attention** and **Feedforward Networks (FFN)**. Self-attention allows every token to weigh the relevance of every other token. Cross-attention is used in encoder-decoder models to connect to the encoder's context.
*   **Training & Variants:** Models can be **encoder-only** (e.g., BERT, for representation learning), **decoder-only** (e.g., GPT, for text generation), or **encoder-decoder**. Innovations include **FlashAttention** for efficient GPU implementation and **RoPE** for relative positional encodings.

## Generative Adversarial Networks (GANs)
GANs define a zero-sum game between two networks: a **Generator** ($\mu_G$) that creates synthetic data, and a **Discriminator** ($D$) that distinguishes between real ($\mu_{ref}$) and generated data.

The objective in the original GAN game is to find a Nash equilibrium where the generator matches the reference distribution. GANs are implicit generative models, meaning they do not explicitly model the likelihood.
*   **Training Challenges:** Standard GANs suffer from unstable convergence and **mode collapse** (where the generator only produces a subset of possible outputs).
*   **Stabilization Techniques:**
    *   **Wasserstein GAN (WGAN):** Replaces the Jensen-Shannon divergence with the Wasserstein-1 distance, providing smoother gradients and mitigating vanishing gradients, achieved by constraining the discriminator's Lipschitz norm (via weight clipping or spectral normalization).
    *   **Conditional GAN (cGAN):** Allows conditional sample generation by feeding labels to both networks.
    *   **InfoGAN:** Encourages the generator to utilize an informative part of the latent space by maximizing the mutual information between a latent code part ($c$) and the generated output.
    *   **CycleGAN:** Enables image-to-image translation between two domains without requiring paired training data, enforced by a cycle consistency loss.
    *   **Progressive GAN (and StyleGAN):** Stabilizes training for high-resolution image generation by progressively increasing the network's resolution during training.

## Convolutional Neural Networks (CNNs)
CNNs are designed to exploit the spatial locality and translation invariance inherent in grid-structured data like images, mitigating the "curse of dimensionality" faced by traditional MLPs.

Key architectural components and features:
*   **Convolutional Layers:** Apply small, shared-weight filters (kernels) across the input to produce feature maps. Parameter sharing drastically reduces complexity.
*   **Pooling Layers:** Downsample feature maps (using max or average pooling) to reduce size, computation, and introduce local translational invariance.
*   **Fully Connected Layers:** Used at the end of the network for final classification based on the processed features.
*   **Receptive Field:** The area in the input that a neuron "sees"; stacking layers increases the effective receptive field.
*   **Regularization:** Techniques like **Dropout** (randomly deactivating nodes) and **Weight Decay** (penalizing large weights) are used to prevent overfitting.

CNNs are used across diverse fields, including image recognition, video analysis, and medical imaging.