## Variational Autoencoders (VAEs)
VAEs are generative models that leverage an amortized approach using neural networks, contrasting with traditional expectation-maximization methods. The model comprises two parts: an **encoder** neural network that maps data points to parameters of a variational distribution in the low-dimensional latent space, and a **decoder** neural network that maps from this latent space back to the input space.

Optimization requires minimizing two terms derived from the free energy: the **reconstruction error** (often mean squared error or cross-entropy) and the **Kullback–Leibler divergence (KL-D)**, which forces the approximate posterior ($q$) to overlap with the assumed prior ($p$). This optimization is achieved by maximizing the **Evidence Lower Bound (ELBO)**, which can be reformulated as minimizing reconstruction error minus the KL-D between $q$ and $p$. The **reparameterization trick** is employed to enable gradient flow through the stochastic latent variable sampling process.

## Generative Adversarial Networks (GANs) and Variants
GANs are based on a zero-sum game between a **generator** ($\mu_G$) and a **discriminator** ($D$). The generator aims to produce samples indistinguishable from the reference distribution ($\mu_{ref}$), while the discriminator aims to correctly classify real vs. generated data.

Variants address GAN limitations: **Wasserstein GANs (WGANs)** use the Wasserstein metric to provide smoother gradients, mitigating mode collapse. **Conditional GANs (CGANs)** allow conditional sample generation using labels. Further advancements include **CycleGAN** for unpaired image-to-image translation and **InfoGAN**, which maximizes the mutual information between a latent variable and the generated output.

## Transformer Models
Transformers revolutionized sequence modeling by replacing recurrence with self-attention, enabling parallel computation over entire input sequences. The basic **encoder-decoder** structure processes input tokens into contextualized representations, which the decoder iteratively generates.

Key components include **tokenizers** (e.g., BPE) and **embedding layers**, which convert text into vectors. **Self-attention** calculates token relationships via scaled dot-product attention, while **multihead attention** allows the model to capture diverse types of relevance. Architectural variations include **encoder-only** (BERT for understanding) and **decoder-only** (GPT for generation) models. Efficiency improvements are achieved via **FlashAttention** and **Swin Transformer**'s windowed attention.

## Graph Neural Networks (GNNs)
GNNs are designed to operate on graph-structured data by implementing fundamental layers: **permutation equivariant** message passing, **local pooling** for downsampling, and **global pooling** (readout) for fixed-size representation. Message passing neural networks (MPNNs) update node features ($\mathbf{h}_u$) by aggregating messages ($\psi$) from neighbors.

Specific architectures include **Graph Convolutional Networks (GCNs)**, a spectral approximation of convolutions for graphs, and **Graph Attention Networks (GATs)**, which use attention coefficients ($\alpha_{uv}$) to weigh neighbor importance. GNNs are being explored for complex tasks like protein folding and network anomaly detection, though their expressivity is limited by the Weisfeiler–Leman Test.

## Autoencoders (AEs)
An autoencoder consists of an encoder ($E_\phi: \mathcal{X} \to \mathcal{Z}$) and a decoder ($D_\theta: \mathcal{Z} \to \mathcal{X}$), trained to minimize the reconstruction error $L(\theta, \phi) = \mathbb{E}_{x \sim \mu_{ref}}[d(x, D_\theta(E_\phi(x)))]$.

Variations include **Variational Autoencoders (VAEs)**, which impose a probabilistic structure on the latent space; **Sparse Autoencoders (SAEs)**, which enforce sparsity in codes; and **Denoising Autoencoders (DAEs)**, which learn robust representations by training on corrupted inputs. **Contractive Autoencoders (CAEs)** further enforce robustness by penalizing the Jacobian matrix of the encoder.

## Convolutional Neural Networks (CNNs)
CNNs are designed for grid-like data (like images), overcoming the curse of dimensionality of traditional MLPs by exploiting spatial locality. Their core building blocks include **convolutional layers**, which apply shared, small-receptive-field kernels to generate feature maps, and **pooling layers** (max or average) for dimensionality reduction and introducing local translational invariance.

CNNs utilize **ReLU** as a primary activation function, and the overall structure is finalized by **fully connected layers** for classification. Regularization techniques like **Dropout** and **weight decay** are crucial for preventing overfitting. Techniques like **dilation** allow for expanding the receptive field without increasing parameters.

## Recurrent Neural Networks (RNNs)
RNNs are characterized by recurrent connections that allow the hidden state ($h_t$) to retain a "memory" of past inputs ($x_t$), making them suitable for sequential data. Standard RNNs suffer from vanishing/exploding gradients over long sequences.

Modern solutions include **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit), which use gating mechanisms to control information flow and maintain long-term dependencies. Architectures include **Bidirectional RNNs (BRNNs)**, which process sequences forwards and backward, and **Encoder-Decoder** configurations used in sequence transduction (like machine translation). Training methods include **Teacher Forcing** and **Backpropagation Through Time (BPTT)**.