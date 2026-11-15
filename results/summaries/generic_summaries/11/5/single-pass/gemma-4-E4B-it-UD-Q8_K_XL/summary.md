## Radial Basis Function (RBF) Networks

RBF networks typically employ a three-layer architecture: an input layer, a hidden layer using a non-linear RBF activation function, and a linear output layer. The network's output $\varphi(\mathbf{x})$ is a sum of weighted radial basis functions centered at $\mathbf{c}_i$. A common function is the Gaussian, which is radially symmetric around its center. RBF networks are universal approximators, meaning enough hidden neurons can approximate any continuous function on a compact set.

RBF networks can be normalized, resulting in a normalized radial basis function $u(\cdot)$, which has theoretical justification in stochastic data flow modeling where the output is the expected value of $y$ given input $\mathbf{x}$. Furthermore, the architecture can be extended to include local linear models by adding weight terms $\mathbf{b}_i \cdot (\mathbf{x} - \mathbf{c}_i)$ to the output.

## Autoencoders (AEs)

An autoencoder is a model consisting of an encoder $E_{\phi}: \mathcal{X} \to \mathcal{Z}$ and a decoder $D_{\theta}: \mathcal{Z} \to \mathcal{X}$. Training aims to minimize a loss function $L(\theta, \phi) = \mathbb{E}_{x \sim \mu_{ref}}[d(x, D_{\theta}(E_{\phi}(x)))]$. Undercomplete AEs compress data, while overcomplete ones might learn the identity function.

Variations include **Variational Autoencoders (VAEs)**, which optimize the Evidence Lower Bound (ELBO) by approximating the true posterior with $q_{\phi}(z|x)$; **Sparse Autoencoders (SAEs)**, which enforce sparsity in the codes; **Denoising Autoencoders (DAEs)**, which are trained by intentionally corrupting the input; and **Contractive Autoencoders (CAEs)**, which regularize the encoder to resist infinitesimal input perturbations.

## Recurrent Neural Networks (RNNs)

RNNs are neural networks designed to process sequential data by maintaining an internal "memory" ($\mathbf{h}_t$). Modern architectures include **Long Short-Term Memory (LSTM)** and **Bidirectional RNNs (BRNN)**, which overcome the vanishing gradient problem inherent in standard RNNs.

Configurations include **Stacked RNNs** (deep RNNs) and **Encoder-Decoder** configurations used for sequence transduction. Training typically uses **Backpropagation Through Time (BPTT)**, though variants like **Real-Time Recurrent Learning (RTRL)** offer online updates. The **Transformer** architecture revolutionized sequence modeling by replacing recurrence with attention mechanisms, enabling parallel computation.

## Graph Neural Networks (GNNs)

GNNs process data structured as graphs $G=(V,E)$. The core mechanism is **Message Passing Neural Networks (MPNNs)**, where nodes update their representations by aggregating messages from neighbors. The architecture typically includes permutation equivariant layers, local pooling for coarsening, and a permutation invariant global pooling layer.

Specialized GNNs include **Graph Convolutional Networks (GCNs)**, which approximate a localized spectral filter on graphs, and **Graph Attention Networks (GATs)**, which use attention coefficients ($\alpha_{uv}$) to weigh neighbor importance. Gated Graph Sequence Neural Networks (GGS-NNs) combine GNNs with GRU cells to output sequences.

## Generative Adversarial Networks (GANs)

GANs involve a zero-sum game between a generator ($\mu_G$) aiming to match the reference distribution ($\mu_{ref}$) and a discriminator ($D$) trying to distinguish real from generated data. The original GAN objective minimizes the Jensen-Shannon divergence, which suffers from vanishing gradients when distributions are disjoint.

To address this, **Wasserstein GANs (WGANs)** replace the objective with the Wasserstein distance, providing a smoother, more informative critic gradient. Variants include **Conditional GANs (CGANs)**, which allow generation based on labels, and **CycleGAN**, which performs unsupervised domain translation using cycle consistency loss. Architectures like **StyleGAN** leverage progressive growth for stable, high-resolution image generation.

## Transformers

Transformers utilize self-attention mechanisms to capture long-range dependencies in sequences in parallel, overcoming the sequential bottleneck of RNNs. Core components involve **tokenizers** (using subword methods like BPE), **embedding** and **positional encoding**, and stacked **Transformer layers** (encoder/decoder).

The **Multihead Attention** mechanism calculates attention scores via scaled dot-products $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$. Training is often performed via **Pretrain-finetune** strategies on large corpora. Key optimizations include **KV caching** for inference, **FlashAttention** for GPU efficiency, and **RoPE** and **ALiBi** for advanced positional encoding.

## Vision Transformers (ViT)

ViTs adapt the Transformer architecture to computer vision. They process an image by dividing it into square **patches**, transforming these patches into vectors via **patch embedding**, adding **positional encoding**, and feeding the resulting sequence through a BERT-like encoder-only Transformer.

Variants include **Swin Transformer**, which uses shifted window attention to mimic CNN receptive fields, and **Masked Autoencoders (MAE)**, which train via self-supervision by reconstructing masked patches. **DINO** uses self-distillation techniques to learn representations without labels.

## Other Key Models

**Hopfield Networks** are associative memory models that converge to local energy minima. **Dense Associative Memories** generalize these to achieve super-linear memory capacity via higher-order interactions. **Convolutional Neural Networks (CNNs)** exploit spatial locality through shared weights and receptive fields, utilizing **max** or **average pooling** for downsampling. **Variational Autoencoders (VAEs)** provide a probabilistic generative framework by optimizing the ELBO.